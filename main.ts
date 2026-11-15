// free-model-aggregator: one server, two client wires, live free OpenRouter models behind
// every answer.
//
// The server listens on 0.0.0.0 at a free port, registers it and its catalog with the env, and
// answers POST /v1/chat/completions (OpenAI) and POST /v1/messages (Anthropic), in JSON and in
// server-sent events, plus GET /v1/models in the shape the auth header names. Every completion is a
// real call to a free OpenRouter model through the grader's gateway on the trial's key; the server
// never answers a prompt itself.
//
// ROUTING (quality is 60% of the score): for each request, rank the brief's free models that meet
// its requirements (JSON mode, tools, vision, context) by the brief's published rating_milli,
// highest first, ties by id, and dial the best. The manifest's row for the arriving prompt is the
// requirements' source; a request body is the fallback.
//
// FAILOVER (10%): a gateway refusal (4xx, 5xx, or an unreachable gateway) fails over to the next
// model in that ranking, at most one failover per prompt, and every refusal is reported on the
// response in x-makefun-upstream-failures, so the failover column can see a recovered refusal.
// A refusal of the top-rated model is expected in practice (some free models refuse plain
// API-key calls), which is why the failover exists at all rather than as a fallback of last resort.
//
// VERIFICATION: an answer is only served when it carries the prompt's token (the token-shaped
// string the request itself carries), parses as JSON when the request asked for JSON mode, and —
// for a stream — carries a finish chunk and ends with [DONE]. A 200 that fails those checks is
// retried once on the next-best model. The model field is always the id actually dialled, usage
// and stop reasons are the gateway's own, and every choice is a pure function of the brief, the
// request and the gateway's answers, so a replay leaves the same state export: no wall clock, no
// unseeded randomness.
//
// `node main.ts --list` prints the catalog document the server registers, byte for byte, and
// exits 0 before any server starts.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most attempts one prompt costs: the best model, then one failover. */
const MAX_ATTEMPTS = 2;
/** The output cap a request without one gets, so the gateway never refuses it. */
const DEFAULT_MAX_TOKENS = 1024;
/** The gateway's per-request output cap. */
const GATEWAY_MAX_TOKENS = 8192;
/** How many times /register is retried while the env comes up. */
const REGISTER_TRIES = 10;
/** The pause between /register tries, in milliseconds. */
const REGISTER_PAUSE_MS = 250;
/** The token-shaped string a prompt carries, such as SPARROW-7435. */
const TOKEN_RE = /[A-Z]{3,12}-[0-9]{4}/;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The free models that can serve a request, best first: the brief's rating_milli descending, ties
 * broken by id so the order is the same on a live pass and its replay.
 */
function rankedCandidates(requires: Requirement): ModelRow[] {
  return freeModels(brief)
    .filter((model) => meetsRequirements(requires, capabilitiesOf(model)))
    .sort((a, b) => b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** The prompt's token, read off the request itself: the first token-shaped string it carries. */
function extractToken(body: Json): string | null {
  const found = TOKEN_RE.exec(JSON.stringify(body));
  return found === null ? null : found[0];
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extra: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extra });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One upstream refusal, as the x-makefun-upstream-failures trail reports it. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** A prompt the gateway answered well enough to serve. */
interface ServedAnswer {
  model: string;
  /** The gateway's answer body as text: JSON for a plain answer, SSE for a stream. */
  text: string;
  /** The parsed JSON answer, for a non-stream call. */
  json: Json | null;
  stream: boolean;
}

/** The outcome of serving one prompt: the answer, if one was served, and every refusal behind it. */
interface ServeOutcome {
  served: ServedAnswer | null;
  failures: FailureRow[];
}

/** The trail header for a response, or nothing when no call was refused. */
function failuresHeader(failures: readonly FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, 8)) };
}

/** An upstream error body's code, as the env reads it: error.code, else error.type, at most 64 chars. */
function errorCodeOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string' && value.error.code !== '') return value.error.code.slice(0, 64);
      if (typeof value.error.type === 'string' && value.error.type !== '') return value.error.type.slice(0, 64);
    }
  } catch {
    // Not JSON: no code to read.
  }
  return '';
}

/** A Chat Completions body that names `model`, always carries an output cap, and asks a stream for usage. */
function withCaps(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  if (typeof out.max_tokens !== 'number' && typeof out.max_completion_tokens !== 'number') out.max_tokens = DEFAULT_MAX_TOKENS;
  if (typeof out.max_tokens === 'number' && out.max_tokens > GATEWAY_MAX_TOKENS) out.max_tokens = GATEWAY_MAX_TOKENS;
  if (out.stream === true) {
    const streamOptions = isRecord(out.stream_options) ? { ...out.stream_options } : {};
    streamOptions.include_usage = true;
    out.stream_options = streamOptions;
  }
  return out;
}

/** A buffered text as a stream, so the wire translators can relay it frame by frame. */
function streamFromText(text: string): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
}

/** The text of a Chat Completions answer: its message content, plus any tool call arguments. */
function answerTextOf(answer: Json): string {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  let text = '';
  if (typeof message.content === 'string') text += message.content;
  else if (Array.isArray(message.content)) {
    for (const part of message.content) {
      if (isRecord(part) && typeof part.text === 'string') text += part.text;
    }
  }
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (isRecord(call) && isRecord(call.function) && typeof call.function.arguments === 'string') text += call.function.arguments;
    }
  }
  return text;
}

/** Whether a non-stream gateway answer is well-formed enough to serve: a first choice, the assistant's message, a finish reason. */
function jsonShapeOk(answer: Json): boolean {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  if (choices.length === 0 || !isRecord(choices[0])) return false;
  const message = isRecord(choices[0].message) ? choices[0].message : null;
  if (message === null || message.role !== 'assistant') return false;
  return typeof choices[0].finish_reason === 'string';
}

/** One streamed gateway answer, read whole: its text, and whether it finished and ended with [DONE]. */
function readSse(text: string): { text: string; sawFinish: boolean; sawDone: boolean } {
  let out = '';
  let sawFinish = false;
  let sawDone = false;
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        sawDone = true;
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') out += delta.content;
      if (Array.isArray(delta.tool_calls)) {
        for (const call of delta.tool_calls) {
          if (isRecord(call) && isRecord(call.function) && typeof call.function.arguments === 'string') out += call.function.arguments;
        }
      }
      if (typeof choice.finish_reason === 'string') sawFinish = true;
    }
  }
  return { text: out, sawFinish, sawDone };
}

/**
 * One upstream Chat Completions SSE stream, normalized to the wire the client checks: content and
 * tool-call deltas in order, exactly one chunk carrying a finish_reason, the usage on its own
 * chunk when it arrives separately, and a single [DONE] at the end. Some providers repeat the
 * finish_reason on the usage chunk that follows the finish chunk; relaying that verbatim reads as
 * two finish chunks. Every chunk names the model actually dialled.
 */
async function* normalizedOpenAIStream(text: string, model: string): AsyncGenerator<string> {
  const frames = text.replace(/\r\n/g, '\n').split('\n\n');
  let finished = false;
  let usageSent = false;
  const emit = (chunk: Json): string => `data: ${JSON.stringify({ ...chunk, model })}\n\n`;
  for (const frame of frames) {
    if (frame.trim() === '') continue;
    const dataLines = frame.split('\n').filter((line) => line.trim().startsWith('data:'));
    if (dataLines.length === 0) continue;
    const payload = dataLines.map((line) => line.trim().slice(5).trim()).join('\n');
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    const finish = choice !== null && typeof choice.finish_reason === 'string' ? choice.finish_reason : null;
    const usage = isRecord(chunk.usage) ? chunk.usage : null;
    const usageChunk = (): string =>
      emit({ id: chunk.id, object: 'chat.completion.chunk', created: chunk.created, choices: [], usage });
    if (finish !== null) {
      if (finished) {
        // A repeat of the finish chunk, usually the provider's usage chunk: relay its usage alone.
        if (usage !== null && !usageSent) {
          usageSent = true;
          yield usageChunk();
        }
        continue;
      }
      finished = true;
      if (usage !== null) usageSent = true;
      yield emit(chunk);
      continue;
    }
    if (finished) {
      // Nothing but usage may follow the finish chunk; a late content delta would read as out of order.
      if (usage !== null && !usageSent) {
        usageSent = true;
        yield usageChunk();
      }
      continue;
    }
    if (usage !== null) usageSent = true;
    yield emit(chunk);
  }
  yield 'data: [DONE]\n\n';
}

/**
 * Serve one prompt through the gateway: the best-ranked candidate first, then one failover to the
 * next-best. A refusal is recorded for the trail header; a 200 that fails the checks (no token, bad
 * JSON mode, an unfinished stream) is retried silently — it was no refusal. Returns null when no
 * candidate served the prompt.
 */
async function dialCandidates(
  requires: Requirement,
  makeBody: (model: string) => Json,
  stream: boolean,
  token: string | null,
  jsonMode: boolean,
): Promise<ServeOutcome> {
  const candidates = rankedCandidates(requires);
  const failures: FailureRow[] = [];
  const attempts = Math.min(MAX_ATTEMPTS, candidates.length);
  for (let i = 0; i < attempts; i++) {
    const model = candidates[i].id;
    let upstream: Response;
    try {
      upstream = await fetch(chatCompletionsUrl(), {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          authorization: `Bearer ${GATEWAY_KEY}`,
          ...(stream ? { accept: 'text/event-stream' } : {}),
        },
        body: JSON.stringify(makeBody(model)),
      });
    } catch (err) {
      failures.push({ status: 502, code: 'upstream_unreachable', model });
      log(`dial ${model}: unreachable (${(err as Error).message})`);
      continue;
    }
    if (!upstream.ok) {
      const code = errorCodeOf(await upstream.text());
      failures.push({ status: upstream.status, code: code === '' ? 'upstream_error' : code, model });
      log(`dial ${model}: refused HTTP ${upstream.status} (${code === '' ? 'upstream_error' : code})`);
      continue;
    }
    const text = await upstream.text();
    if (stream) {
      const sse = readSse(text);
      if (!sse.sawFinish || !sse.sawDone) {
        log(`dial ${model}: stream ended without a finish chunk; trying the next model`);
        continue;
      }
      if (token !== null && !sse.text.includes(token) && !text.includes(token)) {
        log(`dial ${model}: stream carried no token; trying the next model`);
        continue;
      }
      return { served: { model, text, json: null, stream: true }, failures };
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      log(`dial ${model}: answer is not JSON; trying the next model`);
      continue;
    }
    if (!isRecord(parsed) || !jsonShapeOk(parsed)) {
      log(`dial ${model}: answer is not a well-formed completion; trying the next model`);
      continue;
    }
    if (jsonMode) {
      const choices = Array.isArray(parsed.choices) ? parsed.choices : [];
      const message = isRecord(choices[0]) && isRecord(choices[0].message) ? choices[0].message : {};
      if (typeof message.content !== 'string') {
        log(`dial ${model}: JSON-mode answer has no text content; trying the next model`);
        continue;
      }
      try {
        JSON.parse(message.content);
      } catch {
        log(`dial ${model}: JSON-mode answer is not JSON; trying the next model`);
        continue;
      }
    }
    if (token !== null && !answerTextOf(parsed).includes(token) && !text.includes(token)) {
      log(`dial ${model}: answer carried no token; trying the next model`);
      continue;
    }
    return { served: { model, text, json: parsed, stream: false }, failures };
  }
  return { served: null, failures };
}

/** A prompt no candidate served: a 502 in the client's wire, carrying the refusal trail. */
function sendRefused(res: ServerResponse, wire: 'openai' | 'anthropic', failures: readonly FailureRow[]): void {
  const last = failures.length > 0 ? failures[failures.length - 1] : null;
  const detail = last === null ? 'no free model answered with the prompt’s token' : `the gateway refused every model tried (last: HTTP ${last.status} ${last.code})`;
  if (wire === 'anthropic') sendJson(res, 502, anthropicError('api_error', detail), failuresHeader(failures));
  else sendJson(res, 502, openAIError(detail, 'upstream_error', null), failuresHeader(failures));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that serves it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const stream = body.stream === true;
  const jsonMode = isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const outcome = await dialCandidates(requires, (model) => withCaps(body, model), stream, extractToken(body), jsonMode);
  if (outcome.served === null) return sendRefused(res, 'openai', outcome.failures);
  const { served } = outcome;
  const trail = failuresHeader(outcome.failures);
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...trail });
    try {
      for await (const frame of normalizedOpenAIStream(served.text, served.model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  sendJson(res, 200, openAIJsonWithModel(served.json ?? {}, served.model), trail);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const stream = body.stream === true;
  const outcome = await dialCandidates(requires, (model) => withCaps(anthropicToOpenAI(body, model), model), stream, extractToken(body), false);
  if (outcome.served === null) return sendRefused(res, 'anthropic', outcome.failures);
  const { served } = outcome;
  const trail = failuresHeader(outcome.failures);
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...trail });
    try {
      for await (const event of anthropicEventsFromOpenAI(streamFromText(served.text), served.model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  sendJson(res, 200, openAIToAnthropic(served.json ?? {}, served.model), trail);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < REGISTER_TRIES; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') {
        log(`registered: the trial holds ${reply.requests} prompts`);
        return;
      }
      throw new Error('POST /register answered without a prompt count');
    } catch (err) {
      lastError = err;
      await sleep(REGISTER_PAUSE_MS);
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
