# Free-Model Aggregator

One server, two API wires, live free models behind every answer.

Point any OpenAI- or Anthropic-compatible client at it and every completion is a real
call to a free OpenRouter model, dialled through the gateway on the key the operator
pushed to the competition. Built for the make.fun competition
**free-model-aggregator-02** ("The best live free-model aggregator").

## Run

```sh
node main.ts            # serves POST /v1/chat/completions, POST /v1/messages, GET /v1/models
node main.ts --list     # prints the catalog document the server registers, then exits 0
```

Node 22, standard library only — nothing is installed. The trial sets
`MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`, `MAKEFUN_TRIAL_SEED`,
`OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `OPENAI_API_KEY` and
`ANTHROPIC_API_KEY`; the server reads the brief (the free-model list with the
house's ratings, and the six prompts' manifest) from `MAKEFUN_PROMPT_FILE`.

## How it routes

- **Quality first.** For each request, the free models that meet its
  requirements (JSON mode, tools, vision, context length) are ranked by the
  brief's published `rating_milli`, highest first, ties broken by id. The best
  model serves the prompt — the strongest JSON-capable model for JSON mode,
  the strongest tools-capable model for a tools prompt, the strongest overall
  otherwise.
- **Failover.** A gateway refusal (4xx, 5xx, or an unreachable gateway) fails
  over to the next-best model that meets the prompt, at most one failover per
  prompt, and every refusal is reported on the response in the
  `x-makefun-upstream-failures` header.
- **Verification before serving.** An answer is only served when it carries
  the prompt's token (read off the request itself), parses as JSON when the
  request asked for JSON mode, and — for a stream — carries a finish chunk and
  ends with `[DONE]`. A 200 that fails those checks is retried once on the
  next-best model. Nothing is ever invented: the `model` field is always the
  id actually dialled, and usage and stop reasons are the gateway's own.
- **Deterministic.** Every choice is a pure function of the brief, the
  request and the gateway's answers — no wall clock, no unseeded randomness —
  so a replay leaves the same state export.
- **Catalog.** The registered catalog and both `GET /v1/models` answers list
  exactly the fourteen dialable free ids, once each.

## Layout

- `main.ts` — the server: the brief, the catalog, the routes, the routing /
  failover / verification, the register and the work poll.
- `lib/providers.ts` — the brief's types and parser, the free model list, the
  catalog builder.
- `lib/wire.ts` — Anthropic ↔ Chat Completions translation, the model field,
  the SSE writers.

## License

MIT — see `LICENSE`.
