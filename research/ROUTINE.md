# ROUTINE.md — Contract for `codking-technique-scout`

This file is the **contract** for the bisemanal Cowork routine that scouts new ML/security techniques for codking. The scheduled agent reads this file **first** on every tick and follows it exactly. Do not invent fields, sources, or sections that are not defined here.

---

## 1. Purpose

Each tick produces:

1. An updated `research/backlog.json` of candidate papers, scored and sorted.
2. **At most one** new viability doc in `research/candidates/`, modeled on `MHC_INTEGRATION.md` at the repo root.
3. A run log in `research/log/`.

The routine never modifies production code, configs, or tests.

---

## 2. Sources

Query arxiv via `WebFetch` against the public API:

```
http://export.arxiv.org/api/query?search_query=cat:cs.LG+OR+cat:cs.AI+OR+cat:cs.CR+OR+cat:cs.CL&start=0&max_results=200&sortBy=submittedDate&sortOrder=descending
```

Filter the response to entries with `submittedDate` within the last **14 days** (the cadence window). If the arxiv endpoint is unreachable, skip Phase 1 and proceed to Phase 2 using the existing backlog. Log the failure in the run log.

---

## 3. Keyword filter

A candidate qualifies for scoring if its **title or abstract** contains at least one of the following terms (case-insensitive, whole-word match preferred):

- `efficient attention`, `sparse attention`, `linear attention`, `flash attention`
- `state space`, `mamba`, `SSM`
- `hierarchical reasoning`, `recurrent reasoning`, `adaptive computation`
- `byte-level`, `tokenizer-free`, `dynamic chunking`, `boundary detection`
- `hyper-connection`, `residual connection`, `manifold`, `sinkhorn`
- `recursive language model`, `infinite context`, `long context`
- `threat detection`, `intrusion detection`, `malware classification`, `anomaly detection`, `log analysis`

A candidate is rejected outright if the abstract describes only a benchmark/dataset release with no architectural contribution.

---

## 4. Scoring rubric (sum to 10)

Compute a numeric score per candidate. Record **both** the total and a breakdown.

| # | Criterion | Max | Notes |
|---|---|---|---|
| 1 | Architectural compatibility with HRM / H-Net / Mamba-2 | 2.5 | Drop-in modification scores higher than wholesale rewrite. |
| 2 | Parameter / latency overhead vs codking targets (27M params, < 10ms) | 2.0 | Hard constraint: > 30% overhead caps this criterion at 0.5. |
| 3 | Empirical evidence quality | 1.5 | Ablations + reported numbers + code released → full score. Code released alone adds +0.5. |
| 4 | Training stability claims | 1.5 | Especially relevant for byte-level / 22-layer Main Network. |
| 5 | Security-domain relevance | 1.5 | Direct hits (threat / anomaly / byte stream) cap rubric at ≥ 9.0. |
| 6 | Recency & momentum | 1.0 | Last 90 days + replications/citations → full score. |

**Threshold to produce a viability doc**: total score ≥ **7.0**. Below 7.0 stays in backlog.

---

## 5. Backlog format (`research/backlog.json`)

```json
{
  "version": 1,
  "updated": "ISO-8601 timestamp of last tick",
  "entries": [
    {
      "arxiv_id": "2512.24880",
      "title": "Manifold-Constrained Hyper-Connections",
      "authors": ["Zhenda Xie", "..."],
      "submitted": "2025-12-29",
      "abstract_url": "https://arxiv.org/abs/2512.24880",
      "matched_keywords": ["hyper-connection", "manifold"],
      "score": 9.0,
      "score_breakdown": {
        "compat": 2.4,
        "overhead": 1.8,
        "evidence": 1.4,
        "stability": 1.5,
        "security": 0.9,
        "recency": 1.0
      },
      "seen_at": "2026-05-04T14:00:00Z",
      "doc_status": "pending"
    }
  ]
}
```

`doc_status` values: `pending` (no doc yet), `produced` (doc in `research/candidates/`), `rejected` (score dropped below threshold or paper retracted).

**Backlog cap**: 20 entries. On overflow, drop the lowest-scored `pending` entry first, never a `produced` one.

**Sort**: by `score` descending, then `submitted` descending.

---

## 6. Dedupe procedure (mandatory before adding to backlog)

For each new candidate `arxiv_id`:

1. `grep -l "<arxiv_id>"` against:
   - `research/backlog.json`
   - `research/candidates/*.md`
   - `*_INTEGRATION.md` at the repo root (e.g. `MHC_INTEGRATION.md`, `RLM_INTEGRATION.md`)
2. If any hit → skip. Do not duplicate.
3. Also check the `slug` for collision in `research/candidates/`.

---

## 7. Naming convention

```
research/candidates/YYYY-MM-DD_<arxiv_id>_<slug>.md
research/log/YYYY-MM-DD_run.md
```

- `YYYY-MM-DD` = date the tick produced the file (UTC).
- `<arxiv_id>` = canonical arxiv id (e.g. `2512.24880`), no dot variations.
- `<slug>` = lowercase, hyphenated, derived from the title, max **30 chars**, no leading/trailing hyphen.

If `research/candidates/<...same name...>.md` already exists, abort Phase 2 for this tick (do not overwrite).

---

## 8. Viability doc template

Every doc in `research/candidates/` MUST follow this top-level structure (modeled on `MHC_INTEGRATION.md` at the repo root):

```markdown
# <Technique Name> Integration Viability Analysis for CodKing

**Paper**: [<title>](https://arxiv.org/abs/<arxiv_id>)
**Authors**: <authors>
**Date**: <paper publication month/year>
**Analysis Date**: <today, YYYY-MM-DD>

---

## Executive Summary

**Viability Assessment: <verdict> (<score>/10)**

<2–4 sentence summary of what this technique does and whether it should
be integrated. Include the headline benefit and the headline cost.>

---

## 1. Technical Analysis

### 1.1 Problem Solved
### 1.2 Proposed Solution
### 1.3 Quantified Benefits

(Reproduce a benefits table modeled on `MHC_INTEGRATION.md` §1.3 if the
paper provides comparable numbers. Otherwise omit the table and state
"benchmarks not directly comparable" with a one-line justification.)

---

## 2. Compatibility with CodKing

State explicitly which subsystems it touches: `models/hnet/`, `models/hrm/`,
`models/integration/`, `models/mhc/`, `rlm_framework/`, training pipeline,
inference path. For each, mark "drop-in", "patch", "rewrite", or "N/A".

---

## 3. Integration Points

For every concrete integration point, cite codking source as
`<relative_path>:<line>`. **Use LSP** (`workspaceSymbol`,
`goToDefinition`, `findReferences`) to obtain these — never grep.
This is mandated by `CLAUDE.md` at the repo root.

---

## 4. Implementation Plan

Numbered step list. Each step: file(s) to add or modify, estimated
delta in LOC, new test file path under `tests/`, new config path
under `configs/` (if applicable). Do NOT actually create any of those
files in this tick — the doc is a proposal, not an implementation.

---

## 5. Risks & Open Questions

Bullet list. Cap at 6 items.

---

## 6. Roadmap

Three phases: PoC, integration, validation. One paragraph each, with
a rough effort estimate (S / M / L).
```

If the paper does not provide enough information for any required
section, write `N/A — <one-line justification>`. Never fabricate
benchmarks, file:line citations, or claims not in the source.

---

## 9. Run log template (`research/log/YYYY-MM-DD_run.md`)

```markdown
# Run YYYY-MM-DD

## Phase 1 — Backlog refresh
- arxiv query: <url>
- raw entries returned: <N>
- entries matching keyword filter: <N>
- entries new (post-dedupe): <N>
- entries added to backlog: <N>
- backlog size after: <N> / 20
- errors: <none | description>

## Phase 2 — Viability doc
- top candidate: <arxiv_id> — <title> (score <X>)
- doc produced: <path | "skipped — top score below 7.0">
- LSP citations resolved: <N>

## Notes
<anything unusual>
```

---

## 10. Guardrails (hard limits — refuse to violate)

The routine MUST:

- Write only under `research/`.
- Never modify `models/`, `configs/`, `tests/`, `api/`, `integrations/`, `rlm_framework/`, `scripts/`, `training/`, or any `*_INTEGRATION.md` at the repo root.
- Produce at most **1** new candidate doc per tick.
- Never overwrite an existing file in `research/candidates/`.
- Never run `pytest`, `./run_tests.sh`, `scripts/train.py`, or the FastAPI server.
- Never execute Python code that imports from `models/` or `rlm_framework/`.
- Cite codking `file:line` only via LSP.
- Stop with a clear log entry on any of: arxiv outage, malformed `backlog.json`, write failure under `research/`.

If a future revision needs to relax any of these limits, edit this file first and bump `version` in `backlog.json`.
