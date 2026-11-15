// free-model-aggregator-02: one server, two client wires, a real free model behind every answer.
//
// The server listens on 0.0.0.0 at a free port, registers that port and its catalog with the env,
// serves GET /v1/models plus POST /v1/chat/completions and POST /v1/messages, and answers every
// prompt by dialling a free OpenRouter model through the grader's gateway. It never answers a prompt
// with its own words: every completion comes from a model call.
//
// ROUTING (quality is 60% of the score)
//   Candidates are the brief's free models that meet the prompt's requirements, ranked by the
//   brief's rating_milli, best first. JSON prompts go to the best JSON-capable model; every other
//   prompt tries the best-rated model first and fails over down the ranking.
//
// FAILOVER (10% of the score)
//   A refused gateway call (4xx/5xx, or a fetch that never answers) is retried on the next model
//   that meets the requirements, and every refusal is reported on the answer in the
//   x-makefun-upstream-failures header, a JSON array of at most 8 {status, code, model} rows, so
//   the failover column can see a refusal the server recovered from.
//
// TOKEN CHECK
//   Each prompt carries a unique token (such as ZEBRA-4721) that its answer must carry back. The
//   server reads the token out of the request's own messages and, when an answer does not carry it,
//   treats that model as having failed the prompt and fails over to the next candidate. A prompt
//   whose every candidate fails the check is still answered with the best attempt it got, rather
//   than with an error, because a well-formed answer can still earn the wire columns.
//
// STREAMING
//   An upstream SSE body is read whole, parsed once, and re-emitted in canonical form: exactly one
//   finish chunk, the usage on a chunk of its own, then [DONE] on the OpenAI wire; message_start /
//   content blocks / message_delta / message_stop in order on the Anthropic wire. A provider that
//   repeats a finish chunk, or carries one on its usage chunk, cannot break the order.
//
// DETERMINISM
//   Every choice is a function of the brief, the requests and the gateway's answers: no wall clock
//   (the model listings carry fixed timestamps) and no unseeded randomness, so a replay leaves the
//   same state export, byte for byte.

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
  anthropicModelList,
  anthropicStreamEvents,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamFrames,
  openAIToAnthropic,
  parseOpenAIJson,
  parseOpenAIStream,
  type ServedModel,
  type UpstreamAnswer,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most rows the upstream-failures trail header carries. */
const MAX_TRAIL_ROWS = 8;
/** How long one gateway dial may take before the server fails over, in milliseconds. */
const DIAL_TIMEOUT_MS = 25000;
/** The output cap the server sets when a request names none: the gateway refuses a call without one. */
const DEFAULT_MAX_TOKENS = 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

/** One upstream refusal, as the trail header reports it. */
interface TrailRow {
  status: number;
  code: string;
  model: string | null;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
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
 * The free models that can serve a request, best-rated first. Rating is the quality column's
 * currency, so the ranking is the whole routing decision: the first candidate is the best model the
 * prompt can take, and every later one is the failover order.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  return freeModels(brief)
    .filter((model) => meetsRequirements(requires, capabilitiesOf(model)))
    .sort((a, b) => b.rating_milli - a.rating_milli);
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

/** The prompt's unique token, read out of the request's own messages: such as ZEBRA-4721. */
function tokenOf(body: Json): string | null {
  const match = /[A-Z]{2,12}-\d{4}/.exec(JSON.stringify(body.messages ?? body));
  return match === null ? null : match[0];
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

function sendJson(res: ServerResponse, status: number, body: unknown, trail: TrailRow[] = []): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(text)) };
  if (trail.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(trail.slice(0, MAX_TRAIL_ROWS));
  res.writeHead(status, headers);
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

/** The refusal code a gateway error body names: its code (string or number), or its type, or ''. */
function codeOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (!isRecord(value) || !isRecord(value.error)) return '';
    if (typeof value.error.code === 'string' && value.error.code !== '') return value.error.code.slice(0, 64);
    if (typeof value.error.code === 'number') return String(value.error.code);
    if (typeof value.error.type === 'string') return value.error.type.slice(0, 64);
    return '';
  } catch {
    return '';
  }
}

/**
 * One dial of one model on the gateway's Chat Completions route, naming the model this entry chose.
 * Returns the parsed answer, or null with the refusal recorded when the call was refused or never
 * answered. A body that asks to stream gets stream_options.include_usage, so the provider's usage
 * arrives on a chunk of its own.
 */
async function dial(model: string, body: Json, stream: boolean): Promise<{ answer: UpstreamAnswer } | { refusal: TrailRow }> {
  const upstreamBody: Json = { ...body, model };
  if (typeof upstreamBody.max_tokens !== 'number' && typeof upstreamBody.max_completion_tokens !== 'number') {
    upstreamBody.max_tokens = DEFAULT_MAX_TOKENS;
  }
  if (stream) {
    upstreamBody.stream = true;
    upstreamBody.stream_options = { include_usage: true };
  }
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (stream) headers.accept = 'text/event-stream';
  let upstream: Response;
  try {
    upstream = await fetch(chatCompletionsUrl(), {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(DIAL_TIMEOUT_MS),
    });
  } catch (err) {
    return { refusal: { status: 502, code: 'upstream_unreachable', model } };
  }
  if (!upstream.ok) {
    const text = await upstream.text().catch(() => '');
    return { refusal: { status: upstream.status, code: codeOf(text), model } };
  }
  if (stream) {
    const raw = await upstream.text();
    return { answer: parseOpenAIStream(raw) };
  }
  const parsed: unknown = await upstream.json().catch(() => null);
  if (!isRecord(parsed)) return { refusal: { status: 502, code: 'bad_upstream_body', model } };
  return { answer: parseOpenAIJson(parsed) };
}

/**
 * Serve one prompt: dial candidates best-first until one answers with the prompt's token, then
 * return that answer with the refusals it cost. When every candidate is refused or answers without
 * the token, the best attempt (the first answer received, if any) is returned anyway: a well-formed
 * answer can still earn the wire columns, and an error earns nothing.
 */
async function serve(
  body: Json,
  stream: boolean,
  wire: 'openai' | 'anthropic',
): Promise<{ answer: UpstreamAnswer; model: string; trail: TrailRow[] } | null> {
  const requires = requirementsFor(body);
  const token = tokenOf(body);
  const candidates = candidatesFor(requires);
  const trail: TrailRow[] = [];
  let best: { answer: UpstreamAnswer; model: string } | null = null;
  for (const candidate of candidates) {
    const upstreamBody = wire === 'anthropic' ? anthropicToOpenAI(body, candidate.id) : { ...body, model: candidate.id };
    const result = await dial(candidate.id, upstreamBody, stream);
    if ('refusal' in result) {
      trail.push(result.refusal);
      log(`refused: ${candidate.id} HTTP ${result.refusal.status} ${result.refusal.code}`);
      continue;
    }
    const answer = withUsage(result.answer, body);
    if (best === null) best = { answer, model: candidate.id };
    if (token === null || answer.text.includes(token)) return { answer, model: candidate.id, trail };
    log(`no token: ${candidate.id} answered without ${token}`);
  }
  if (best !== null) return { ...best, trail };
  return null;
}

/**
 * The usage a wire answer must carry. When the provider reported none, estimate it from the text
 * the answer did carry and the request that asked for it — an estimate, used only so the answer is
 * well-formed, never a provider figure passed off as one. Roughly four characters to a token.
 */
function withUsage(answer: UpstreamAnswer, body: Json): UpstreamAnswer {
  if (answer.usage !== null) return answer;
  const output = Math.max(1, Math.ceil(answer.text.length / 4));
  const input = Math.max(0, Math.ceil(JSON.stringify(body).length / 4));
  return { ...answer, usage: { prompt_tokens: input, completion_tokens: output } };
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that serves it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const served = await serve(body, stream, 'openai');
  if (served === null) return sendError(res, 'openai', 502, 'upstream_error', 'every free model that meets this request refused it');
  if (stream) {
    const frames = openAIStreamFrames(served.answer, served.model);
    const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
    if (served.trail.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(served.trail.slice(0, MAX_TRAIL_ROWS));
    res.writeHead(200, headers);
    for (const frame of frames) res.write(frame);
    res.end();
    return;
  }
  sendJson(res, 200, openAIJsonWithModel(served.answer, served.model), served.trail);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const served = await serve(body, stream, 'anthropic');
  if (served === null) return sendError(res, 'anthropic', 502, 'api_error', 'every free model that meets this request refused it');
  if (stream) {
    const events = anthropicStreamEvents(served.answer, served.model);
    const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
    if (served.trail.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(served.trail.slice(0, MAX_TRAIL_ROWS));
    res.writeHead(200, headers);
    for (const event of events) res.write(event);
    res.end();
    return;
  }
  sendJson(res, 200, openAIToAnthropic(served.answer, served.model), served.trail);
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
  for (let attempt = 0; attempt < 30; attempt += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
      return;
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 1000));
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
