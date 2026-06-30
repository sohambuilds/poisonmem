# Persistent-memory retrieval-poisoning sanity check

For the full research rationale, operating procedure, decision rules, troubleshooting, and
collaborator checklist, see [GUIDE.md](GUIDE.md).

This repository implements the controlled pilot in the supplied design suite. It tests only the
path from pre-seeded procedural records, through fixed-budget retrieval, to a later simulated action.
Every tool name is an enum; the harness never executes a real deployment, payment, publication, or
data export.

## Registered design choices

- Models: `openai/gpt-4o-mini` and `meta-llama/llama-4-scout` through OpenRouter.
- Pinned providers: `openai` and `deepinfra`, respectively; `only`, `order`, and
  `allow_fallbacks: false` are sent on every request.
- Shared response mode: `json_object`. Change `openrouter.response_mode` to `json_schema` only after
  `validate-providers` confirms both pinned endpoints support it.
- Retrieval budget: every method returns at most four records.
- RRF candidate depth: the union of each base retriever's top 12, using `1 / (60 + rank)`.
- Agreement: a record must be in the top 12 of at least two base retrievers, then is ranked by RRF.
- Oracle clean contexts: four clean BM25 matches. Oracle contexts replace rank two with the matching
  domain poison and retain three clean matches.
- Control drift: unique control queries newly unsafe under either poisoned RRF bank, relative to the
  same model's clean-RRF result.
- Confidence intervals: 2,000 paired bootstrap samples clustered by query ID with seed `20260627`.

These choices resolve details that were not fixed in the handoff while keeping all primary context
budgets and decision thresholds unchanged.

## Setup

The project uses `uv` exclusively:

```bash
uv sync
uv run memory-sanity init-data
uv run memory-sanity validate-data
```

Stage C is local but downloads the two registered embedding checkpoints on first use:

```bash
uv run memory-sanity retrieval
```

For OpenRouter stages, set the credential in the environment. Do not commit it:

```bash
export OPENROUTER_API_KEY='...'
uv run memory-sanity validate-providers
```

## Execution

The paid call counts are 384 for preflight, 384 for oracle influence, and 2,880 for the full matrix.
The end-to-end command is resumable and requires explicit cost confirmation:

```bash
uv run memory-sanity run --confirm-paid-calls
```

It runs local retrieval and endpoint validation, executes Stage A, and stops before poisoning stages
when the validity gate fails. A valid preflight proceeds through Stages B and D and then writes the
fixed verdict. Each logical repetition has a stable call ID, so rerunning the command skips completed
records in `results/generations.jsonl`.

If Stage A fails, inspect the recorded failures, make at most one harness repair, and rerun preflight.
The run fingerprint includes the data, prompt source, model/provider configuration, and generation
settings, so records from the pre-repair protocol are retained for audit but excluded from the new
gate. Do not interpret poisoning stages from a failed preflight.

Stages can also be run separately:

```bash
uv run memory-sanity preflight --confirm-paid-calls
uv run memory-sanity oracle --confirm-paid-calls
uv run memory-sanity full --confirm-paid-calls
uv run memory-sanity analyze
```

Only malformed model output is retried, and at most once. HTTP failures are recorded without an
automatic model retry. OpenRouter generation metadata is fetched to capture provider, request ID,
usage, cost, timestamp, and prompt hash.

## Outputs

The fixed inputs are generated under `data/`. A completed run creates:

```text
results/
  retrieval_results.parquet
  generations.jsonl
  api_usage.jsonl
  metrics.json
  decision.json
  sanity_report.md
```

`decision.json` is produced mechanically from the registered thresholds. No LLM judge or subjective
override participates in scoring.

## Development checks

```bash
uv run ruff check .
uv run pytest
```
