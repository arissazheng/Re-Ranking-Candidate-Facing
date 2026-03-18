# Candidate Search Re-Ranking Pipeline

A retrieval and re-ranking system for searching ~200K LinkedIn profiles indexed in Turbopuffer, evaluated against Mercor's private scoring endpoint.

## Approach

The core insight driving this pipeline is that **hard criteria are multiplicative gates**: a candidate who fails even one hard criterion scores 0, regardless of soft criteria quality. This means the pipeline must prioritize hard criteria pass rates above all else, and only then optimize for soft criteria.

The system works in three phases:

### Phase 1: Cast a wide net with targeted retrieval
Rather than relying on a single embedding query, the pipeline runs **multiple semantically distinct queries** per role (e.g., for Doctors: one query emphasizing "family medicine physician from top medical school" and another emphasizing "primary care telemedicine EHR experience"). Each query is paired with both strict and broad Turbopuffer metadata filters, and results are merged and deduplicated. This typically surfaces 300–500 unique candidates per role.

### Phase 2: Aggressively filter on hard criteria using structured data
Each candidate's `degrees` and `experience` strings are parsed into structured dicts to enable precise rule-based checks. For each of the 10 public query configs, a custom hard filter function verifies domain-specific requirements:

- **Degree type validation** — Distinguishes actual MDs from PhDs in biomedical engineering by checking field of study against positive (clinical medicine) and negative (engineering, physics) pattern lists
- **School quality tiering** — Evaluator-calibrated whitelists for medical schools, law schools, and universities, built from observed pass/fail patterns across multiple evaluation runs
- **Title matching** — Catches false positives like "postdoctoral" matching "doctor" by excluding known problematic substrings
- **Recency verification** — For the Anthropology query, a dual-gate system requires both structured metadata (start year ≥ 2023) AND evaluator-visible text evidence (phrases like "first-year student" or "started PhD in 2024"), because the evaluator reads profile text and ignores structured fields

### Phase 3: LLM re-ranking with domain-specific guidance
The top 50–75 candidates (after filtering and heuristic pre-sorting) are sent to GPT-4o with full structured education/experience data and query-specific evaluation instructions. The LLM scores each candidate 0–10, with explicit guidance on what constitutes passing for ambiguous hard criteria (e.g., "top U.S. medical school" means Harvard/Hopkins/Stanford/Duke-tier, not any US school).

### Handling unseen queries
For private/unseen query configs, the pipeline falls back to LLM-based filter extraction: GPT-4o-mini analyzes the query's hard criteria to dynamically extract degree requirements, country filters, experience minimums, field-of-study keywords, and title patterns. This builds both Turbopuffer filters and a rule-based hard filter function on the fly, so private queries get the same structured pipeline as public ones.

## Architecture

```
Query Config
    │
    ├─ Voyage AI embed (2-6 queries per role)
    │       │
    │       ▼
    ├─ Turbopuffer ANN search (strict + broad filters)
    │       │
    │       ▼  300-500 candidates
    ├─ Parse degrees/experience structured strings
    │       │
    │       ▼
    ├─ Rule-based hard filter (per-query custom logic)
    │       │
    │       ▼  50-400 candidates
    ├─ Domain-specific pre-sort (school tier, board cert, etc.)
    │       │
    │       ▼  top 50-75
    ├─ GPT-4o re-ranking (batched, with query-specific guidance)
    │       │
    │       ▼  scored 0-10
    └─ Submit top 10 to evaluation endpoint
```

## Results

Best scores achieved across runs (baseline → final):

| Query | Baseline | Best |
|-------|----------|------|
| Bankers | 25.3 | 90.7 |
| Mathematics PhD | 7.5 | 90.0 |
| Mechanical Engineers | 76.3 | 89.7 |
| Biology Expert | 15.0 | 86.0 |
| Quantitative Finance | 17.0 | 84.3 |
| Tax Lawyer | N/A | 83.7 |
| Radiology | 60.0 | 78.7 |
| Jr. Corporate Lawyer | 57.7 | 78.7 |
| Doctors (MD) | 8.0 | 76.5 |
| Anthropology | 8.3 | 60.3 |

## Files

- `pipeline.py` — Main pipeline: retrieval, filtering, re-ranking, evaluation
- `query_configs.py` — The 10 public evaluation query definitions
- `requirements.txt` — Python dependencies
