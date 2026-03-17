from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import turbopuffer
import voyageai
from openai import OpenAI

from query_configs import QueryConfig, get_all_query_configs


TPUF_NAMESPACE = "search-test-v4"


@dataclass
class Candidate:
    object_id: str
    score: float
    data: Dict[str, Any]


def make_tpuf_client() -> turbopuffer.Turbopuffer:
    api_key = os.environ.get("TURBOPUFFER_API_KEY")
    if not api_key:
        raise RuntimeError("TURBOPUFFER_API_KEY environment variable is required.")
    return turbopuffer.Turbopuffer(api_key=api_key, region="aws-us-west-2")


def make_voyage_client() -> voyageai.Client:
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY environment variable is required.")
    return voyageai.Client(api_key=api_key)


def make_openai_client() -> OpenAI:
    api_key = os.environ.get("OAI_KEY")
    if not api_key:
        raise RuntimeError("OAI_KEY environment variable is required.")
    return OpenAI(api_key=api_key)


def llm_rewrite_query_and_extract_filters(
    client: OpenAI, query: QueryConfig
) -> Tuple[str, Dict[str, Any]]:
    """Use an LLM to (a) rewrite the query as an ideal candidate bio and (b) extract simple filters.

    The filters are intentionally constrained so we can reliably map them to Turbopuffer / profile fields.
    """
    system = (
        "You help build candidate search over structured LinkedIn profiles.\n"
        "Given a role description, hard criteria, and soft criteria, you must:\n"
        "1) Write an ideal candidate bio as if it were their LinkedIn summary.\n"
        "2) Extract a small set of structured filters that can be applied to profile metadata.\n\n"
        "Return a strict JSON object with keys:\n"
        "  ideal_bio: string\n"
        "  filters: {\n"
        "    required_degrees: string[]         # e.g. ['JD', \"Doctorate\", \"MBA\", \"MD\", \"Master's\", \"Bachelor's\"]\n"
        "    required_degree_countries: string[] # e.g. ['United States', 'India', 'Canada', 'United Kingdom']\n"
        "    min_total_experience_years: number | null\n"
        "    min_relevant_experience_years: number | null\n"
        "    required_title_keywords: string[]   # e.g. ['Attorney', 'Tax Lawyer']\n"
        "    required_country_residence: string[] # profile country field, e.g. ['United States']\n"
        "  }\n"
        "Only include things that are clearly required by the hard criteria; do not guess.\n"
        "If something is not required, set the corresponding value to null or an empty list.\n"
    )
    user = {
        "role_name": query.name,
        "description": query.description,
        "hard_criteria": query.hard_criteria,
        "soft_criteria": query.soft_criteria,
    }
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
        temperature=0.1,
    )
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("Empty response from OpenAI when rewriting query.")
    obj = json.loads(content)
    ideal_bio = obj.get("ideal_bio", "").strip()
    filters = obj.get("filters") or {}
    return ideal_bio, filters


def build_tpuf_filters(structured_filters: Dict[str, Any]) -> Optional[List[Any]]:
    """Convert structured filters into Turbopuffer filter expression.

    We keep this intentionally simple and focused on obvious, high-signal constraints.
    """
    clauses: List[Any] = []

    required_degrees = structured_filters.get("required_degrees") or []
    if required_degrees:
        # deg_degrees is an array; use ContainsAny to ensure at least one required degree
        clauses.append(["deg_degrees", "ContainsAny", required_degrees])

    degree_countries = structured_filters.get("required_degree_countries") or []
    # The schema doesn't expose degree country separately; ignore for now.

    country_residence = structured_filters.get("required_country_residence") or []
    if country_residence:
        clauses.append(["country", "In", country_residence])

    # Years of experience: profiles expose various bucketed fields; we enforce total
    # experience loosely by checking that exp_years contains sufficiently large buckets.
    min_total_years = structured_filters.get("min_total_experience_years")
    if isinstance(min_total_years, (int, float)) and min_total_years >= 3:
        # If they want 3+ years, require at least the '3' bucket; for 5+ or 10+ you could extend this logic.
        bucket = "3" if min_total_years <= 3 else "5" if min_total_years <= 5 else "10"
        clauses.append(["exp_years", "ContainsAny", [bucket]])

    if not clauses:
        return None
    if len(clauses) == 1:
        # Single filter, no And wrapper needed.
        attr, op, value = clauses[0]
        return [attr, op, value]
    return ["And", clauses]


def retrieve_candidates(
    query: QueryConfig,
    ideal_bio: str,
    structured_filters: Dict[str, Any],
    tpuf_client: turbopuffer.Turbopuffer,
    voyage_client: voyageai.Client,
    top_k: int = 150,
) -> List[Candidate]:
    """Initial retrieval from Turbopuffer using vector ANN + metadata filters."""
    ns = tpuf_client.namespace(TPUF_NAMESPACE)

    embed_text = ideal_bio or query.description
    emb_resp = voyage_client.embed(embed_text, model="voyage-3")
    embeddings = emb_resp.embeddings[0]

    tpuf_filters = build_tpuf_filters(structured_filters)

    result = ns.query(
        rank_by=("vector", "ANN", embeddings),
        top_k=top_k,
        include_attributes=True,
        filters=tpuf_filters,
    )

    candidates: List[Candidate] = []
    for row in result.rows:
        # turbopuffer Row objects expose id, score, attributes
        row_id = getattr(row, "id", None) or getattr(row, "_id", None)
        if not row_id:
            continue
        score = getattr(row, "score", 0.0)
        data = getattr(row, "attributes", {}) or {}
        candidates.append(
            Candidate(object_id=str(row_id), score=float(score or 0.0), data=data)
        )
    return candidates


def passes_hard_criteria_rule_based(
    candidate: Candidate, query: QueryConfig, structured_filters: Dict[str, Any]
) -> bool:
    """Apply additional rule-based checks for hard criteria using structured fields.

    This is intentionally conservative: we only accept when we can clearly verify requirements.
    """
    data = candidate.data

    # Degree checks based on deg_degrees
    required_degrees = structured_filters.get("required_degrees") or []
    if required_degrees:
        degs = {d.lower() for d in (data.get("deg_degrees") or [])}
        if not any(req.lower() in degs for req in required_degrees):
            return False

    # Country residency
    required_countries = structured_filters.get("required_country_residence") or []
    if required_countries:
        country = (data.get("country") or "").strip()
        if country and country not in required_countries:
            return False

    # Minimal total experience as an approximate guardrail
    min_total_years = structured_filters.get("min_total_experience_years")
    if isinstance(min_total_years, (int, float)) and min_total_years >= 3:
        buckets = set(data.get("exp_years") or [])
        if not (("3" in buckets) or ("5" in buckets) or ("10" in buckets)):
            return False

    # For certain roles, add simple title-based guards.
    title_keywords = [t.lower() for t in (structured_filters.get("required_title_keywords") or [])]
    if title_keywords:
        titles = [t.lower() for t in (data.get("exp_titles") or [])]
        if not any(any(kw in t for kw in title_keywords) for t in titles):
            return False

    return True


def llm_rerank_candidates(
    client: OpenAI, query: QueryConfig, candidates: Iterable[Candidate], max_candidates: int = 40
) -> List[Candidate]:
    """LLM-based re-ranking using rerankSummary and the full query definition."""
    cands = list(candidates)[:max_candidates]
    if not cands:
        return []

    items = []
    for idx, c in enumerate(cands):
        summary = c.data.get("rerankSummary") or c.data.get("rerank_summary") or ""
        titles = ", ".join(c.data.get("exp_titles") or [])
        degrees = ", ".join(c.data.get("deg_degrees") or [])
        item = {
            "index": idx,
            "id": c.object_id,
            "name": c.data.get("name", ""),
            "country": c.data.get("country", ""),
            "degrees": degrees,
            "titles": titles,
            "summary": summary,
        }
        items.append(item)

    system = (
        "You are ranking candidates for a role based on their LinkedIn-like profiles.\n"
        "You will be given a role description, hard criteria, and soft criteria, and a list of candidate profiles.\n"
        "For each candidate, assign a relevance score from 0 to 10 where:\n"
        "  - First, if they clearly fail hard criteria, they must get a score <= 3.\n"
        "  - Among remaining candidates, higher scores mean better match on both hard and soft criteria.\n"
        "Be decisive. Use the full 0-10 range.\n"
        "Return a JSON object with key 'scores', a list of {id, score} objects.\n"
    )
    user = {
        "role_name": query.name,
        "description": query.description,
        "hard_criteria": query.hard_criteria,
        "soft_criteria": query.soft_criteria,
        "candidates": items,
    }
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
        temperature=0.1,
    )
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("Empty response from OpenAI during re-ranking.")
    obj = json.loads(content)
    scores_raw = obj.get("scores") or []
    scores: Dict[str, float] = {}
    for s in scores_raw:
        cid = str(s.get("id"))
        try:
            scores[cid] = float(s.get("score"))
        except (TypeError, ValueError):
            continue

    # Merge scores back into candidates
    scored: List[Candidate] = []
    for c in cands:
        if c.object_id in scores:
            scored.append(Candidate(object_id=c.object_id, score=scores[c.object_id], data=c.data))

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored


def call_evaluation_endpoint(
    config_path: str,
    object_ids: List[str],
    auth_email: str,
    base_url: str = "https://mercor-dev--search-eng-interview.modal.run/evaluate",
) -> Dict[str, Any]:
    if not object_ids:
        raise ValueError("object_ids must be non-empty.")
    payload = {"config_path": config_path, "object_ids": object_ids}
    headers = {
        "Content-Type": "application/json",
        "Authorization": auth_email,
    }
    resp = requests.post(base_url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def run_pipeline_for_query(
    query: QueryConfig,
    tpuf_client: turbopuffer.Turbopuffer,
    voyage_client: voyageai.Client,
    openai_client: OpenAI,
    auth_email: Optional[str] = None,
    submit: bool = False,
    top_k_submit: int = 10,
) -> Dict[str, Any]:
    """End-to-end pipeline for a single query: retrieval, filtering, LLM re-rank, optional evaluation."""
    ideal_bio, structured_filters = llm_rewrite_query_and_extract_filters(openai_client, query)

    initial_candidates = retrieve_candidates(
        query=query,
        ideal_bio=ideal_bio,
        structured_filters=structured_filters,
        tpuf_client=tpuf_client,
        voyage_client=voyage_client,
    )

    filtered_candidates = [
        c for c in initial_candidates if passes_hard_criteria_rule_based(c, query, structured_filters)
    ]

    reranked = llm_rerank_candidates(openai_client, query, filtered_candidates)
    top = reranked[:top_k_submit] if reranked else []
    top_ids = [c.object_id for c in top]

    eval_result: Optional[Dict[str, Any]] = None
    if submit and auth_email:
        eval_result = call_evaluation_endpoint(query.config_path, top_ids, auth_email)

    return {
        "query_name": query.name,
        "config_path": query.config_path,
        "ideal_bio": ideal_bio,
        "structured_filters": structured_filters,
        "num_initial": len(initial_candidates),
        "num_after_hard_filter": len(filtered_candidates),
        "top_ids": top_ids,
        "evaluation": eval_result,
    }


def run_all_queries(submit: bool = False) -> List[Dict[str, Any]]:
    """Run the full pipeline for all 10 public configs."""
    tpuf_client = make_tpuf_client()
    voyage_client = make_voyage_client()
    openai_client = make_openai_client()

    auth_email = os.environ.get("MERCOR_EVAL_EMAIL")
    if submit and not auth_email:
        raise RuntimeError("MERCOR_EVAL_EMAIL environment variable is required when submit=True.")

    results: List[Dict[str, Any]] = []
    for cfg in get_all_query_configs():
        res = run_pipeline_for_query(
            cfg,
            tpuf_client=tpuf_client,
            voyage_client=voyage_client,
            openai_client=openai_client,
            auth_email=auth_email,
            submit=submit,
        )
        results.append(res)
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Mercor re-ranking pipeline over all 10 configs.")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="If set, will submit top-10 IDs for each config to the evaluation endpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Path to write a JSON summary of results.",
    )
    args = parser.parse_args()

    results = run_all_queries(submit=args.submit)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote pipeline results for {len(results)} queries to {args.output}")
    for r in results:
        print(
            f"{r['query_name']} ({r['config_path']}): "
            f"{len(r['top_ids'])} candidates submitted, "
            f"{r['num_initial']} initial / {r['num_after_hard_filter']} after hard filter."
        )


if __name__ == "__main__":
    main()

