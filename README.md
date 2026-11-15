# Free-Model Aggregator

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/Arisaf7/free-model-aggregator-02-entry/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/Arisaf7/free-model-aggregator-02-entry/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

Built for the make.fun bounty [The best live free-model aggregator](https://make.fun/bounty/free-model-aggregator-02) ($500 in USDC).

One server that speaks both the OpenAI and Anthropic wires, and answers every
completion with a real call to a free OpenRouter model, dialled through the
grader's gateway. Built for the make.fun `free-model-aggregator-02` bounty.

## What it does

- `POST /v1/chat/completions` — OpenAI-compatible, JSON or server-sent events
- `POST /v1/messages` — Anthropic-compatible, JSON or server-sent events
- `GET /v1/models` — the model list in the shape the auth header names
  (`Authorization: Bearer` gets the OpenAI list, `x-api-key` the Anthropic one)
- `node main.ts --list` — prints the catalog document and exits 0

## How it routes

Candidates are the brief's free models that meet the prompt's requirements,
ranked by the brief's published rating, best first. JSON prompts go to the
best JSON-capable model; every other prompt tries the best-rated model first
and fails over down the ranking when a call is refused — or when the answer
does not carry the prompt's unique token back.

Every refusal a prompt recovers from is reported on the answer in the
`x-makefun-upstream-failures` header, so the failover is visible to the
grader. Upstream SSE is buffered, parsed once, and re-emitted in canonical
form: exactly one finish chunk, the usage on a chunk of its own, then
`[DONE]` (or the Anthropic `message_start` … `message_stop` order).

## Run it

```sh
node main.ts
```

The trial sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`, `MAKEFUN_TRIAL_SEED`,
`OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `OPENAI_API_KEY` and
`ANTHROPIC_API_KEY`. The server picks a free port, registers it with the env,
serves the six prompts, polls `GET /work`, and exits 0 when the trial is done.

## License

MIT — see `LICENSE`.
