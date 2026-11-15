// The client wires, the one upstream wire, and the model field.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an
// Anthropic one (POST /v1/messages), with a JSON body or with server-sent events. The entry dials
// the grader's gateway, an OpenAI-compatible endpoint, so every prompt goes upstream as a Chat
// Completions request and the Anthropic path is translated both ways.
//
// THE MODEL FIELD
//   The house records every gateway call and scores the model that really answered, so a response's
//   `model` must be the model this entry dialled. OpenRouter's own answer names a free model without
//   the gateway's `openrouter/` prefix; these functions put the dialled id back on every answer and
//   every streamed chunk. They never make a model up: the id came from the brief and went to the
//   gateway in the request's `model`.
//
// STREAMING
//   Upstream SSE is buffered and parsed once into one UpstreamAnswer (text, tool calls, the first
//   finish reason, usage), then re-emitted in canonical form. That is what makes the stream
//   well-formed no matter what the provider sent: exactly one finish chunk, the usage on a chunk of
//   its own, and [DONE] last on the OpenAI wire; message_start / content blocks / message_delta /
//   message_stop in order on the Anthropic wire.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body
// and the same gateway answer give the same bytes on a replay. The two model listings use fixed
// timestamps for the same reason: a `created` that moved between runs would be a nondeterministic
// document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** One tool call the provider made, arguments accumulated across streamed deltas. */
export interface ToolCall {
  id: string;
  name: string;
  /** The arguments as a JSON string, exactly as the provider sent them (possibly in pieces). */
  arguments: string;
}

/** One upstream Chat Completions answer, JSON or streamed, parsed into its parts. */
export interface UpstreamAnswer {
  /** The assistant's text, in order, with streamed deltas joined. */
  text: string;
  /** The tool calls, in the order their first delta arrived. */
  toolCalls: ToolCall[];
  /** The first finish_reason the provider sent, or null when it sent none. */
  finishReason: string | null;
  /** The usage the provider reported, or null when it reported none. */
  usage: { prompt_tokens: number; completion_tokens: number } | null;
  /** The model spelling the provider's own answer carried (bare, without the gateway prefix). */
  providerModel: string;
}

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. `model` is the id the entry dialled. */
export function openAIToAnthropic(answer: UpstreamAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({ type: 'tool_use', id: call.id !== '' ? call.id : 'toolu_unknown', name: call.name, input: argumentsOf(call.arguments) });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(answer.finishReason),
    stop_sequence: null,
    usage: {
      input_tokens: answer.usage === null ? 0 : answer.usage.prompt_tokens,
      output_tokens: answer.usage === null ? 0 : answer.usage.completion_tokens,
    },
  };
}

/** A Chat Completions answer with `model` set to the id the entry dialled: the one truthful claim. */
export function openAIJsonWithModel(answer: UpstreamAnswer, model: string): Json {
  const body: Json = {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: answer.text,
          ...(answer.toolCalls.length > 0
            ? {
                tool_calls: answer.toolCalls.map((call) => ({
                  id: call.id !== '' ? call.id : 'call_unknown',
                  type: 'function',
                  function: { name: call.name, arguments: call.arguments },
                })),
              }
            : {}),
        },
        finish_reason: typeof answer.finishReason === 'string' ? answer.finishReason : 'stop',
      },
    ],
    usage: {
      prompt_tokens: answer.usage === null ? 0 : answer.usage.prompt_tokens,
      completion_tokens: answer.usage === null ? 0 : answer.usage.completion_tokens,
      total_tokens: answer.usage === null ? 0 : answer.usage.prompt_tokens + answer.usage.completion_tokens,
    },
  };
  return body;
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split a raw SSE body into its data payloads, in order; `[DONE]` is returned as the string itself. */
function ssePayloads(raw: string): (Json | '[DONE]')[] {
  const out: (Json | '[DONE]')[] = [];
  const normalized = raw.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '') continue;
      if (payload === '[DONE]') {
        out.push('[DONE]');
        continue;
      }
      try {
        const parsed: unknown = JSON.parse(payload);
        if (isRecord(parsed)) out.push(parsed);
      } catch {
        // Not JSON: not a chunk this parser can use.
      }
    }
  }
  return out;
}

/**
 * Parse a streamed Chat Completions body into one answer. Content deltas join in order; tool-call
 * deltas accumulate per index; the first finish_reason wins (a provider that repeats it, or carries
 * it on the usage chunk, does not make a second finish); usage is the last one the stream carried.
 */
export function parseOpenAIStream(raw: string): UpstreamAnswer {
  let text = '';
  const toolCalls: ToolCall[] = [];
  const byIndex = new Map<number, ToolCall>();
  let finishReason: string | null = null;
  let usage: UpstreamAnswer['usage'] = null;
  let providerModel = '';
  for (const chunk of ssePayloads(raw)) {
    if (chunk === '[DONE]') continue;
    if (typeof chunk.model === 'string' && chunk.model !== '' && providerModel === '') providerModel = chunk.model;
    if (isRecord(chunk.usage)) {
      usage = { prompt_tokens: numberOrZero(chunk.usage.prompt_tokens), completion_tokens: numberOrZero(chunk.usage.completion_tokens) };
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    if (choices.length === 0) continue;
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : byIndex.size;
      let entry = byIndex.get(upstreamIndex);
      if (entry === undefined) {
        entry = { id: '', name: '', arguments: '' };
        byIndex.set(upstreamIndex, entry);
        toolCalls.push(entry);
      }
      if (typeof call.id === 'string' && call.id !== '') entry.id = call.id;
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.name === 'string' && fn.name !== '') entry.name = fn.name;
      if (typeof fn.arguments === 'string') entry.arguments += fn.arguments;
    }
    if (typeof choice.finish_reason === 'string' && finishReason === null) finishReason = choice.finish_reason;
  }
  return { text, toolCalls, finishReason, usage, providerModel };
}

/** Parse a non-streamed Chat Completions body into one answer. */
export function parseOpenAIJson(body: Json): UpstreamAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const toolCalls: ToolCall[] = [];
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({
      id: typeof call.id === 'string' ? call.id : '',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      arguments: typeof call.function.arguments === 'string' ? call.function.arguments : '',
    });
  }
  const usage = isRecord(body.usage)
    ? { prompt_tokens: numberOrZero(body.usage.prompt_tokens), completion_tokens: numberOrZero(body.usage.completion_tokens) }
    : null;
  return {
    text,
    toolCalls,
    finishReason: typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
    usage,
    providerModel: typeof body.model === 'string' ? body.model : '',
  };
}

/**
 * One answer as canonical OpenAI SSE frames: the text as one content chunk (tool calls, when any,
 * as one chunk carrying them), exactly one finish chunk, the usage on a chunk of its own, then
 * [DONE]. `model` is the id the entry dialled, named back on every chunk.
 */
export function openAIStreamFrames(answer: UpstreamAnswer, model: string): string[] {
  const id = 'chatcmpl-aggregate';
  const frames: string[] = [];
  const delta: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    delta.tool_calls = answer.toolCalls.map((call, index) => ({
      index,
      id: call.id !== '' ? call.id : `call_${index}`,
      type: 'function',
      function: { name: call.name, arguments: call.arguments },
    }));
  }
  frames.push(
    sseEvent(null, {
      id,
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: null }],
    }),
  );
  frames.push(
    sseEvent(null, {
      id,
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta: {}, finish_reason: typeof answer.finishReason === 'string' ? answer.finishReason : 'stop' }],
    }),
  );
  if (answer.usage !== null) {
    frames.push(
      sseEvent(null, {
        id,
        object: 'chat.completion.chunk',
        created: FIXED_OPENAI_CREATED,
        model,
        choices: [],
        usage: {
          prompt_tokens: answer.usage.prompt_tokens,
          completion_tokens: answer.usage.completion_tokens,
          total_tokens: answer.usage.prompt_tokens + answer.usage.completion_tokens,
        },
      }),
    );
  }
  frames.push('data: [DONE]\n\n');
  return frames;
}

/**
 * One answer as the Anthropic event sequence: message_start, one text block (then one tool_use
 * block per tool call, with its input as a single input_json_delta), message_delta carrying the stop
 * reason and the output tokens, then message_stop. `model` is the id the entry dialled.
 */
export function anthropicStreamEvents(answer: UpstreamAnswer, model: string): string[] {
  const events: string[] = [];
  const outputTokens = answer.usage === null ? 0 : answer.usage.completion_tokens;
  const inputTokens = answer.usage === null ? 0 : answer.usage.prompt_tokens;
  events.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: inputTokens, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  if (answer.text !== '') {
    events.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: answer.text } }));
    events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of answer.toolCalls) {
    events.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id !== '' ? call.id : `toolu_${index}`, name: call.name, input: {} },
      }),
    );
    if (call.arguments !== '') {
      events.push(
        sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.arguments } }),
      );
    }
    events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  events.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReasonOf(answer.finishReason), stop_sequence: null },
      usage: { output_tokens: outputTokens },
    }),
  );
  events.push(sseEvent('message_stop', { type: 'message_stop' }));
  return events;
}
