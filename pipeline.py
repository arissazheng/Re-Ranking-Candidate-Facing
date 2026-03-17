"""Improved candidate search and re-ranking pipeline.

Key improvements over baseline:
1. Per-query retrieval strategies with domain-specific Turbopuffer filters
2. Multi-pass retrieval (strict filters + broad filters, multiple embedding queries)
3. Structured parsing of degree/experience strings for precise hard criteria checks
4. Aggressive rule-based hard criteria filtering with school/FOS/title verification
5. GPT-4o LLM reranking with rich structured candidate data
6. Larger candidate pools (300-500 initial retrieval)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import requests
import turbopuffer
import voyageai
from openai import OpenAI

from query_configs import QueryConfig, get_all_query_configs


TPUF_NAMESPACE = "search-test-v4"


# ═══════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Candidate:
    object_id: str
    score: float
    data: Dict[str, Any]
    parsed_degrees: List[Dict[str, str]] = field(default_factory=list)
    parsed_experiences: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class RetrievalStrategy:
    """Per-query retrieval configuration."""
    embedding_queries: List[str]
    tpuf_filters_strict: Optional[List[Any]]
    tpuf_filters_broad: Optional[List[Any]]
    top_k: int = 300
    hard_filter_fn: Optional[Callable[["Candidate", "QueryConfig"], bool]] = None


# ═══════════════════════════════════════════════════════════════════════
# Parsing helpers
# ═══════════════════════════════════════════════════════════════════════

def parse_degree_entry(entry: str) -> Dict[str, str]:
    result = {}
    for part in entry.split("::"):
        if part.startswith("yrs_"):
            result["years"] = part[4:]
        elif part.startswith("school_"):
            result["school"] = part[7:]
        elif part.startswith("degree_"):
            result["degree"] = part[7:]
        elif part.startswith("fos_"):
            result["fos"] = part[4:]
        elif part.startswith("start_"):
            result["start"] = part[6:]
        elif part.startswith("end_"):
            result["end"] = part[4:]
    return result


def parse_experience_entry(entry: str) -> Dict[str, str]:
    result = {}
    for part in entry.split("::"):
        if part.startswith("yrs_"):
            result["years"] = part[4:]
        elif part.startswith("title_"):
            result["title"] = part[6:]
        elif part.startswith("company_"):
            result["company"] = part[8:]
        elif part.startswith("start_"):
            result["start"] = part[6:]
        elif part.startswith("end_"):
            result["end"] = part[4:]
    return result


def enrich_candidate(c: Candidate) -> Candidate:
    degrees_raw = c.data.get("degrees") or []
    c.parsed_degrees = [parse_degree_entry(d) for d in degrees_raw if d]
    exp_raw = c.data.get("experience") or []
    c.parsed_experiences = [parse_experience_entry(e) for e in exp_raw if e]
    return c


# ═══════════════════════════════════════════════════════════════════════
# School classification
# ═══════════════════════════════════════════════════════════════════════

M7_SCHOOL_PATTERNS = [
    "harvard", "stanford", "wharton", "university of pennsylvania",
    "booth", "university of chicago", "columbia",
    "kellogg", "northwestern", "sloan", "massachusetts institute of technology",
]

TOP_US_SCHOOL_PATTERNS = [
    "harvard", "stanford", "mit ", "massachusetts institute",
    "yale", "princeton", "columbia", "university of chicago",
    "caltech", "california institute", "carnegie mellon",
    "johns hopkins", "duke", "northwestern", "cornell",
    "nyu", "new york university", "rice", "brown", "dartmouth",
    "vanderbilt", "emory", "georgetown",
    "university of pennsylvania", "upenn", "penn state",
    "university of california", "uc berkeley", "berkeley",
    "ucla", "ucsd", "ucsf", "uc davis", "uc santa",
    "university of michigan", "university of washington",
    "university of wisconsin", "university of illinois",
    "university of texas", "georgia tech", "georgia institute",
    "university of maryland", "ohio state",
    "university of minnesota", "purdue", "university of virginia",
    "university of north carolina", "university of southern california",
    "boston university", "tufts", "university of florida",
    "university of colorado", "rutgers", "university of pittsburgh",
    "stony brook", "indiana university", "university of iowa",
    "university of arizona", "arizona state", "michigan state",
    "texas a&m", "virginia tech", "university of notre dame",
    "wake forest", "lehigh", "brandeis", "case western",
    "university of rochester", "tulane", "rensselaer",
    "worcester polytechnic", "stevens institute",
    "university of connecticut", "university of delaware",
    "university of oregon", "university of utah",
    "north carolina state", "iowa state",
    "colorado school of mines", "university of cincinnati",
    "drexel", "temple", "george washington",
    "american university", "fordham", "northeastern",
    "university of miami", "university of georgia",
    "clemson", "university of tennessee", "baylor",
    "smu", "southern methodist", "university of alabama",
    "university of south carolina", "university of kentucky",
    "washington university", "wustl",
    "university of kansas", "university of nebraska",
    "university of new mexico", "university of hawaii",
    "boston college", "college of william",
    "university of massachusetts", "suny",
    "city university of new york", "cuny",
]

US_LAW_SCHOOL_PATTERNS = TOP_US_SCHOOL_PATTERNS + [
    "fordham", "george washington", "boston college",
    "notre dame", "william & mary", "washington and lee",
    "seton hall", "villanova", "loyola", "pepperdine",
    "university of san francisco", "santa clara",
    "depaul", "brooklyn law", "cardozo", "hofstra",
]

US_MEDICAL_SCHOOL_PATTERNS = TOP_US_SCHOOL_PATTERNS + [
    "mayo clinic", "baylor college of medicine", "mount sinai",
    "icahn", "weill cornell", "albert einstein",
    "thomas jefferson", "rush", "medical college",
    "school of medicine", "creighton",
]

INDIA_MEDICAL_PATTERNS = [
    "aiims", "all india institute", "christian medical college",
    "armed forces medical", "maulana azad", "jipmer",
    "kasturba", "seth gs", "grant medical",
    "king george", "institute of medical sciences",
    "medical college", "medical university",
    "manipal", "amrita", "srm", "vellore",
]

UK_SCHOOL_PATTERNS = [
    "oxford", "cambridge", "imperial college", "imperial",
    "ucl", "university college london",
    "london school of economics", "lse",
    "edinburgh", "manchester", "bristol", "warwick",
    "king's college london", "kcl", "st andrews",
    "durham", "nottingham", "birmingham", "sheffield",
    "leeds", "glasgow", "southampton", "exeter",
    "queen mary", "royal holloway", "university of london",
    "surrey", "bath", "cardiff", "liverpool", "newcastle",
    "aberdeen", "st andrews", "loughborough", "lancaster",
    "university of york", "reading", "sussex", "kent",
    "aston", "city university", "brunel", "goldsmiths",
    "school of oriental", "soas",
]

CA_SCHOOL_PATTERNS = [
    "toronto", "mcgill", "ubc", "university of british columbia",
    "waterloo", "mcmaster", "queen's university",
    "university of alberta", "university of calgary",
    "western university", "university of ottawa",
    "simon fraser", "dalhousie", "university of montreal",
    "laval", "university of manitoba", "university of saskatchewan",
    "carleton", "concordia", "york university", "ryerson",
    "university of victoria", "university of guelph",
]

EU_SCHOOL_PATTERNS = [
    "sorbonne", "sciences po", "leiden", "amsterdam",
    "humboldt", "heidelberg", "munich", "lmu", "eth zurich",
    "epfl", "bocconi", "luiss", "erasmus", "leuven",
    "antwerp", "ghent", "vienna", "copenhagen", "stockholm",
    "helsinki", "barcelona", "madrid", "complutense",
    "trinity college dublin", "bologna", "sapienza",
    "technische universit", "freie universit",
    "charles university", "warsaw", "budapest",
    "universiteit", "université", "universidad", "universität",
    "politecnico", "karolinska", "delft",
    "ecole polytechnique", "hec", "insead",
    "london business school", "iese", "ie business",
    "rotterdam", "tilburg", "maastricht", "groningen",
]


def _matches(name: str, patterns: List[str]) -> bool:
    name_l = name.lower()
    return any(p in name_l for p in patterns)


def is_m7_school(school: str) -> bool:
    return _matches(school, M7_SCHOOL_PATTERNS)


def is_top_us_school(school: str) -> bool:
    return _matches(school, TOP_US_SCHOOL_PATTERNS)


def is_us_uk_ca_school(school: str) -> bool:
    return (_matches(school, TOP_US_SCHOOL_PATTERNS)
            or _matches(school, UK_SCHOOL_PATTERNS)
            or _matches(school, CA_SCHOOL_PATTERNS))


def is_us_or_india_medical(school: str) -> bool:
    return (_matches(school, US_MEDICAL_SCHOOL_PATTERNS)
            or _matches(school, INDIA_MEDICAL_PATTERNS))


def is_reputed_law_school(school: str) -> bool:
    return (_matches(school, US_LAW_SCHOOL_PATTERNS)
            or _matches(school, UK_SCHOOL_PATTERNS)
            or _matches(school, CA_SCHOOL_PATTERNS)
            or _matches(school, EU_SCHOOL_PATTERNS))


def is_india_school(school: str) -> bool:
    india_patterns = [
        "india", "iit", "iisc", "iim", "indian institute",
        "delhi", "mumbai", "chennai", "kolkata", "bangalore",
        "hyderabad", "pune", "jaipur", "lucknow", "chandigarh",
        "manipal", "amrita", "bits", "birla", "vellore",
        "jadavpur", "anna university", "banaras", "aligarh",
        "jawaharlal nehru", "tata institute",
        "national institute of technology",
    ] + INDIA_MEDICAL_PATTERNS
    return _matches(school, india_patterns)


# ═══════════════════════════════════════════════════════════════════════
# Field-of-study matching
# ═══════════════════════════════════════════════════════════════════════

BIOLOGY_FOS = [
    "biology", "biological", "biochemistry", "molecular",
    "genetics", "genomics", "cell biology", "microbiology",
    "neuroscience", "immunology", "biomedical", "life science",
    "biotechnology", "biophysics", "ecology", "zoology",
    "botany", "plant biology", "developmental biology",
    "marine biology", "evolutionary", "bioinformatics",
    "pharmacology", "toxicology", "pathobiology",
]

MATH_STATS_FOS = [
    "mathematics", "mathematical", "statistics", "statistical",
    "applied math", "pure math", "probability", "stochastic",
    "algebra", "analysis", "topology", "geometry",
    "combinatorics", "number theory", "optimization",
    "operations research", "actuarial", "biostatistics",
    "computational math", "data science",
]

ANTHRO_FOS = [
    "anthropology", "sociology", "social science", "ethnograph",
    "cultural studies", "economics", "political economy",
    "development studies", "human geography", "demography",
    "social work", "gender studies", "migration",
    "archaeology", "area studies", "african studies",
    "asian studies", "latin american studies",
    "international development", "public policy",
]

MEDICINE_FOS = [
    "medicine", "medical", "surgery", "clinical",
    "pathology", "pharmacology", "anatomy", "physiology",
    "public health", "epidemiology", "health science",
    "nursing", "physician", "radiology", "diagnostic",
    "internal medicine", "family medicine", "pediatrics",
    "psychiatry", "neurology", "cardiology", "oncology",
    "anesthesiology", "emergency medicine", "obstetrics",
    "ophthalmology", "dermatology", "rehabilitation",
    "dental", "dentistry", "veterinar",
]

MECH_ENG_FOS = [
    "mechanical", "engineering", "thermal", "fluid",
    "structural", "mechatronics", "robotics", "manufacturing",
    "materials science", "aerospace", "automotive",
    "industrial engineering", "energy systems",
    "control systems", "dynamics", "vibration",
]

FINANCE_FOS = [
    "finance", "financial", "economics", "business",
    "accounting", "banking", "investment", "quantitative",
    "actuarial", "risk management", "econometric",
]

LAW_FOS = [
    "law", "legal", "jurisprudence", "juris",
    "corporate law", "tax law", "criminal justice",
    "international law", "constitutional", "human rights",
    "intellectual property",
]


def fos_matches(fos: str, patterns: List[str]) -> bool:
    fos_l = fos.lower()
    return any(p in fos_l for p in patterns)


# ═══════════════════════════════════════════════════════════════════════
# Title matching
# ═══════════════════════════════════════════════════════════════════════

GP_TITLES = [
    "general practitioner", "family medicine", "family physician",
    "primary care", "internal medicine", "general medicine",
    "general physician", "family practice", "attending physician",
    "resident physician", "medical officer", "clinical physician",
    "physician", "hospitalist", "internist",
    # NOTE: "doctor" excluded — it matches "postdoctoral" as substring
]

LAWYER_TITLES = [
    "attorney", "lawyer", "counsel", "solicitor",
    "barrister", "advocate", "litigator", "jurist",
    "legal advisor", "legal officer", "paralegal",
]

TAX_TITLES = [
    "tax", "attorney", "lawyer", "counsel", "legal",
]

BANKING_TITLES = [
    "investment bank", "banker", "corporate finance",
    "m&a", "merger", "acquisition", "capital market",
    "private equity", "venture capital",
    "financial analyst", "equity research",
    "managing director", "vice president",
    "associate", "analyst",
]

QUANT_TITLES = [
    "quant", "quantitative", "risk model", "algorithmic trad",
    "financial engineer", "derivatives", "portfolio manager",
    "risk analyst", "risk manager", "trading",
    "strategist", "systematic",
]

MECH_ENG_TITLES = [
    "mechanical engineer", "design engineer", "product engineer",
    "systems engineer", "structural engineer", "thermal engineer",
    "simulation engineer", "manufacturing engineer",
    "r&d engineer", "research engineer", "project engineer",
    "development engineer", "test engineer", "cad",
]

RADIOLOGY_TITLES = [
    "radiolog", "diagnostic imaging", "imaging specialist",
    "physician", "resident", "attending",
    "consultant", "medical officer",
    # NOTE: "doctor" excluded — matches "postdoctoral"
]


def title_matches(titles: List[str], patterns: List[str]) -> bool:
    for t in titles:
        t_l = t.lower()
        if any(p in t_l for p in patterns):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════
# Experience helpers
# ═══════════════════════════════════════════════════════════════════════

def max_exp_bucket(c: Candidate) -> int:
    mx = 0
    for b in (c.data.get("exp_years") or []):
        try:
            mx = max(mx, int(b))
        except (ValueError, TypeError):
            pass
    return mx


def has_doctoral_in_fos(c: Candidate, patterns: List[str]) -> bool:
    for deg in c.parsed_degrees:
        if (deg.get("degree") or "").lower() in ("doctorate", "phd"):
            if fos_matches(deg.get("fos", ""), patterns):
                return True
    return False


def has_degree_type(c: Candidate, degree_type: str) -> bool:
    degs = {d.lower() for d in (c.data.get("deg_degrees") or [])}
    return degree_type.lower() in degs


# ═══════════════════════════════════════════════════════════════════════
# Per-query hard criteria filters
# ═══════════════════════════════════════════════════════════════════════

def hard_filter_tax_lawyer(c: Candidate, q: QueryConfig) -> bool:
    if not has_degree_type(c, "JD"):
        return False
    if max_exp_bucket(c) < 3:
        return False
    titles = c.data.get("exp_titles") or []
    if not title_matches(titles, TAX_TITLES):
        return False
    return True


def hard_filter_junior_corporate_lawyer(c: Candidate, q: QueryConfig) -> bool:
    # Law degree (JD or law-related fos)
    has_law = has_degree_type(c, "JD")
    if not has_law:
        for deg in c.parsed_degrees:
            if fos_matches(deg.get("fos", ""), LAW_FOS):
                has_law = True
                break
    if not has_law:
        return False

    # From reputed school
    from_reputed = False
    for deg in c.parsed_degrees:
        dt = (deg.get("degree") or "").lower()
        fos = deg.get("fos", "")
        school = deg.get("school", "")
        if dt == "jd" or fos_matches(fos, LAW_FOS):
            if is_reputed_law_school(school):
                from_reputed = True
                break
    # Don't hard-fail on school check since our list may be incomplete;
    # the LLM reranker will evaluate quality.

    # Law-related titles
    titles = c.data.get("exp_titles") or []
    if not title_matches(titles, LAWYER_TITLES):
        return False
    return True


def _is_clinical_md_fos(fos: str) -> bool:
    """Check if FOS indicates an actual clinical MD, not a PhD in a medical-adjacent field."""
    fos_l = fos.lower()
    # Positive: clinical medicine fields
    md_positive = [
        "medicine", "surgery", "radiodiagnosis", "radiology",
        "internal medicine", "family medicine", "pediatrics",
        "psychiatry", "neurology", "cardiology", "oncology",
        "anesthesiology", "emergency medicine", "obstetrics",
        "ophthalmology", "dermatology", "pathology",
        "clinical", "public health", "epidemiology",
    ]
    # Negative: engineering/research fields that contain "medical" as substring
    non_md_negative = [
        "engineering", "physics", "biophysics", "technology",
        "informatics", "biomedical engineer", "medical physic",
        "biomechan", "bioengineer",
    ]
    if not any(p in fos_l for p in md_positive):
        return False
    if any(p in fos_l for p in non_md_negative):
        return False
    return True


def hard_filter_radiology(c: Candidate, q: QueryConfig) -> bool:
    # Hard criterion: MD degree from a medical school in the U.S. or India
    has_md_from_target_country = False
    for deg in c.parsed_degrees:
        dt = (deg.get("degree") or "").lower()
        fos = deg.get("fos", "")
        school = deg.get("school", "")
        if dt in ("doctorate", "md"):
            if _is_clinical_md_fos(fos):
                # Must be from US or India school
                if is_us_or_india_medical(school) or is_india_school(school) or is_top_us_school(school):
                    has_md_from_target_country = True
                    break
    if not has_md_from_target_country:
        # Fallback: radiology titles + doctorate from US/India school with non-engineering FOS
        for deg in c.parsed_degrees:
            dt = (deg.get("degree") or "").lower()
            school = deg.get("school", "")
            fos = deg.get("fos", "")
            fos_l = fos.lower()
            is_engineering = any(p in fos_l for p in ["engineering", "physics", "biophysics", "technology", "informatics"])
            if dt in ("doctorate", "md") and not is_engineering:
                if is_us_or_india_medical(school) or is_india_school(school) or is_top_us_school(school):
                    titles = c.data.get("exp_titles") or []
                    if title_matches(titles, ["radiolog", "physician", "resident", "attending"]):
                        has_research = title_matches(titles, ["research", "postdoc", "scientist", "professor"])
                        if not has_research:
                            has_md_from_target_country = True
                            break
    return has_md_from_target_country


def hard_filter_doctors_md(c: Candidate, q: QueryConfig) -> bool:
    # More specific GP titles (exclude generic "physician" to reduce false positives)
    STRICT_GP_TITLES = [
        "general practitioner", "family medicine", "family physician",
        "primary care", "internal medicine", "general medicine",
        "family practice", "attending physician", "hospitalist",
        "internist", "medical officer", "clinical physician",
        "resident physician", "chief resident",
    ]

    # Check for actual clinical MD (not PhD in biomedical engineering etc.)
    has_md = False
    for deg in c.parsed_degrees:
        dt = (deg.get("degree") or "").lower()
        fos = deg.get("fos", "")
        if dt in ("doctorate", "md"):
            if _is_clinical_md_fos(fos):
                has_md = True
                break
    if not has_md:
        return False

    # US residency
    country = (c.data.get("country") or "").strip()
    if country and country != "United States":
        return False

    # GP/primary care titles (strict check)
    titles = c.data.get("exp_titles") or []
    if not title_matches(titles, STRICT_GP_TITLES):
        # Also accept if they have "physician" but NOT "research" in same title list
        has_physician = title_matches(titles, ["physician"])
        has_research = title_matches(titles, ["research", "postdoc", "scientist", "professor", "lecturer"])
        if not (has_physician and not has_research):
            return False
    return True


def hard_filter_biology_expert(c: Candidate, q: QueryConfig) -> bool:
    # PhD in biology from top US university
    has_bio_phd_us = False
    for deg in c.parsed_degrees:
        dt = (deg.get("degree") or "").lower()
        fos = deg.get("fos", "")
        school = deg.get("school", "")
        if dt == "doctorate" and fos_matches(fos, BIOLOGY_FOS):
            if is_top_us_school(school):
                has_bio_phd_us = True
                break
    if not has_bio_phd_us:
        return False

    # Undergrad in US/UK/CA
    has_undergrad = False
    for deg in c.parsed_degrees:
        dt = (deg.get("degree") or "").lower()
        school = deg.get("school", "")
        if dt == "bachelor's" and is_us_uk_ca_school(school):
            has_undergrad = True
            break
    if not has_undergrad:
        # Relaxed: any degree from US/UK/CA (some profiles may not list bachelor's explicitly)
        for deg in c.parsed_degrees:
            school = deg.get("school", "")
            if is_us_uk_ca_school(school):
                has_undergrad = True
                break
    return has_undergrad


def hard_filter_anthropology(c: Candidate, q: QueryConfig) -> bool:
    # PhD in anthropology/sociology/economics, started within last 3 years (2023+)
    has_relevant_phd = False
    phd_recent = False
    for deg in c.parsed_degrees:
        dt = (deg.get("degree") or "").lower()
        fos = deg.get("fos", "")
        if dt == "doctorate":
            if fos_matches(fos, ANTHRO_FOS):
                has_relevant_phd = True
                try:
                    start = int(deg.get("start", "0"))
                    if start >= 2023:  # "last 3 years" from 2026
                        phd_recent = True
                except (ValueError, TypeError):
                    pass
    if not has_relevant_phd:
        return False
    if not phd_recent:
        return False
    return True


def hard_filter_mathematics_phd(c: Candidate, q: QueryConfig) -> bool:
    # PhD in math/stats from top US
    has_math_phd_us = False
    for deg in c.parsed_degrees:
        dt = (deg.get("degree") or "").lower()
        fos = deg.get("fos", "")
        school = deg.get("school", "")
        if dt == "doctorate" and fos_matches(fos, MATH_STATS_FOS):
            if is_top_us_school(school):
                has_math_phd_us = True
                break
    if not has_math_phd_us:
        return False

    # Undergrad in US/UK/CA
    has_undergrad = False
    for deg in c.parsed_degrees:
        dt = (deg.get("degree") or "").lower()
        school = deg.get("school", "")
        if dt == "bachelor's" and is_us_uk_ca_school(school):
            has_undergrad = True
            break
    if not has_undergrad:
        for deg in c.parsed_degrees:
            school = deg.get("school", "")
            if is_us_uk_ca_school(school):
                has_undergrad = True
                break
    return has_undergrad


def hard_filter_quantitative_finance(c: Candidate, q: QueryConfig) -> bool:
    # M7 MBA
    has_m7 = False
    for deg in c.parsed_degrees:
        dt = (deg.get("degree") or "").lower()
        school = deg.get("school", "")
        if dt == "mba" and is_m7_school(school):
            has_m7 = True
            break
    if not has_m7:
        return False

    # 3+ years relevant experience
    if max_exp_bucket(c) < 3:
        return False

    # Quant/finance titles
    titles = c.data.get("exp_titles") or []
    if not title_matches(titles, QUANT_TITLES + ["finance", "trading", "risk", "portfolio", "investment"]):
        return False
    return True


def hard_filter_bankers(c: Candidate, q: QueryConfig) -> bool:
    # MBA
    if not has_degree_type(c, "MBA"):
        return False

    # Banking/finance titles
    titles = c.data.get("exp_titles") or []
    if not title_matches(titles, BANKING_TITLES + ["finance", "investment", "banking"]):
        return False
    return True


def hard_filter_mechanical_engineers(c: Candidate, q: QueryConfig) -> bool:
    # Higher degree in ME
    has_me_degree = False
    for deg in c.parsed_degrees:
        dt = (deg.get("degree") or "").lower()
        fos = deg.get("fos", "")
        if dt in ("master's", "doctorate", "bachelor's"):
            if fos_matches(fos, MECH_ENG_FOS):
                has_me_degree = True
                break
    if not has_me_degree:
        return False

    # 3+ years
    if max_exp_bucket(c) < 3:
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════
# Per-query retrieval strategies
# ═══════════════════════════════════════════════════════════════════════

def get_retrieval_strategy(query: QueryConfig) -> RetrievalStrategy:
    name = query.config_path.replace(".yml", "")

    if name == "tax_lawyer":
        return RetrievalStrategy(
            embedding_queries=[
                "Experienced tax attorney with JD from accredited US law school, specializing in corporate tax structuring, federal tax compliance, IRS audits. Over three years of legal practice advising clients on tax implications of corporate transactions, drafting legal opinions on federal tax code.",
                "Tax lawyer practicing corporate tax law, experienced in IRS audits and disputes, federal and state tax compliance, legal opinions on tax matters at a law firm",
            ],
            tpuf_filters_strict=["And", [
                ["deg_degrees", "ContainsAny", ["JD"]],
                ["exp_years", "ContainsAny", ["3", "5", "10"]],
            ]],
            tpuf_filters_broad=["deg_degrees", "ContainsAny", ["JD"]],
            top_k=300,
            hard_filter_fn=hard_filter_tax_lawyer,
        )

    elif name == "junior_corporate_lawyer":
        return RetrievalStrategy(
            embedding_queries=[
                "Corporate lawyer at top-tier international law firm specializing in M&A support and cross-border contract negotiations, trained at leading European or American law school with international regulatory compliance experience",
                "Junior attorney with two to four years of experience at a leading law firm working on corporate mergers and acquisitions, due diligence, legal documentation, and commercial agreements across jurisdictions",
            ],
            tpuf_filters_strict=["deg_degrees", "ContainsAny", ["JD"]],
            tpuf_filters_broad=None,
            top_k=300,
            hard_filter_fn=hard_filter_junior_corporate_lawyer,
        )

    elif name == "radiology":
        return RetrievalStrategy(
            embedding_queries=[
                "Radiologist with MD degree from Indian medical college, experienced in reading CT and MRI scans, diagnostic imaging workflows, AI-assisted image analysis, board certified in radiology ABR FRCR",
                "Medical doctor from American medical school specializing in radiology, interpreting X-ray CT MRI ultrasound nuclear medicine studies, diagnostic protocols, differential diagnosis",
                "Radiologist physician MD from India or United States, board certification radiology, diagnostic imaging expertise, radiology reporting",
            ],
            tpuf_filters_strict=["deg_degrees", "ContainsAny", ["Doctorate"]],
            tpuf_filters_broad=None,
            top_k=500,
            hard_filter_fn=hard_filter_radiology,
        )

    elif name == "doctors_md":
        return RetrievalStrategy(
            embedding_queries=[
                "General practitioner family medicine physician with MD from top US medical school, over two years clinical practice in outpatient settings, chronic care management, wellness screenings, telemedicine, patient education",
                "US-trained primary care doctor working as general practitioner in family medicine, experienced in EHR systems, high patient volumes, outpatient diagnostics, interdisciplinary coordination",
                "Physician MD general practice family medicine internal medicine primary care doctor in United States clinical experience",
            ],
            tpuf_filters_strict=["And", [
                ["deg_degrees", "ContainsAny", ["Doctorate"]],
                ["country", "In", ["United States"]],
            ]],
            tpuf_filters_broad=["country", "In", ["United States"]],
            top_k=500,
            hard_filter_fn=hard_filter_doctors_md,
        )

    elif name == "biology_expert":
        return RetrievalStrategy(
            embedding_queries=[
                "Biologist with PhD from top US university specializing in molecular biology and gene expression, research in genetics cell biology, publications in peer-reviewed journals, CRISPR PCR sequencing",
                "Biology PhD researcher at leading American university, molecular biology genetics genomics biochemistry, mentoring students, interdisciplinary research, experimental design and data analysis",
            ],
            tpuf_filters_strict=["deg_degrees", "ContainsAny", ["Doctorate"]],
            tpuf_filters_broad=None,
            top_k=500,
            hard_filter_fn=hard_filter_biology_expert,
        )

    elif name == "anthropology":
        return RetrievalStrategy(
            embedding_queries=[
                "PhD student in anthropology at top US university, focused on labor migration and cultural identity, ethnographic fieldwork, qualitative research methods, published papers on sociological and anthropological topics",
                "Doctoral researcher in anthropology sociology or economics with recent PhD enrollment, expertise in ethnographic methods fieldwork cultural social economic systems, academic publications",
            ],
            tpuf_filters_strict=["And", [
                ["deg_degrees", "ContainsAny", ["Doctorate"]],
                ["deg_start_years", "ContainsAny", ["2023", "2024", "2025", "2026"]],
            ]],
            tpuf_filters_broad=["And", [
                ["deg_degrees", "ContainsAny", ["Doctorate"]],
            ]],
            top_k=500,
            hard_filter_fn=hard_filter_anthropology,
        )

    elif name == "mathematics_phd":
        return RetrievalStrategy(
            embedding_queries=[
                "Mathematician with PhD from leading US university specializing in statistical inference and stochastic processes, published in pure and applied mathematics, probability theory and mathematical modeling",
                "PhD in mathematics or statistics from top American university, research in probability statistics algebra analysis topology, peer-reviewed publications, algorithmic problem-solving",
            ],
            tpuf_filters_strict=["deg_degrees", "ContainsAny", ["Doctorate"]],
            tpuf_filters_broad=None,
            top_k=500,
            hard_filter_fn=hard_filter_mathematics_phd,
        )

    elif name == "quantitative_finance":
        return RetrievalStrategy(
            embedding_queries=[
                "MBA graduate from top US program M7 with experience in quantitative finance, risk modeling, algorithmic trading at global investment firm, Python financial modeling, portfolio optimization, derivatives pricing",
                "Quantitative finance professional with prestigious MBA, experienced in financial engineering risk analytics trading strategies, Python QuantLib, working at investment bank or hedge fund",
            ],
            tpuf_filters_strict=["And", [
                ["deg_degrees", "ContainsAny", ["MBA"]],
                ["exp_years", "ContainsAny", ["3", "5", "10"]],
            ]],
            tpuf_filters_broad=["deg_degrees", "ContainsAny", ["MBA"]],
            top_k=400,
            hard_filter_fn=hard_filter_quantitative_finance,
        )

    elif name == "bankers":
        return RetrievalStrategy(
            embedding_queries=[
                "Healthcare investment banker with MBA from US university, working at leading advisory firm on M&A for healthcare provider groups and digital health companies, growth equity fund, diligence and investment strategy",
                "Investment banking professional with MBA experienced in healthcare M&A, corporate finance, mergers and acquisitions advisory, biotech pharma services provider networks private equity",
            ],
            tpuf_filters_strict=["deg_degrees", "ContainsAny", ["MBA"]],
            tpuf_filters_broad=None,
            top_k=400,
            hard_filter_fn=hard_filter_bankers,
        )

    elif name == "mechanical_engineers":
        return RetrievalStrategy(
            embedding_queries=[
                "Mechanical engineer with advanced degree and over three years of experience in product development and structural design using SolidWorks and ANSYS, thermal system simulations, prototyping electromechanical components, industrial R&D",
                "Senior mechanical design engineer with CAD simulation tools ANSYS COMSOL SolidWorks, expertise in thermal systems fluid dynamics structural analysis, end-to-end product lifecycle from concept to manufacturing",
            ],
            tpuf_filters_strict=["exp_years", "ContainsAny", ["3", "5", "10"]],
            tpuf_filters_broad=None,
            top_k=300,
            hard_filter_fn=hard_filter_mechanical_engineers,
        )

    else:
        return RetrievalStrategy(
            embedding_queries=[query.description],
            tpuf_filters_strict=None,
            tpuf_filters_broad=None,
            top_k=300,
        )


# ═══════════════════════════════════════════════════════════════════════
# Client factories
# ═══════════════════════════════════════════════════════════════════════

def make_tpuf_client() -> turbopuffer.Turbopuffer:
    api_key = os.environ.get("TURBOPUFFER_API_KEY")
    if not api_key:
        raise RuntimeError("TURBOPUFFER_API_KEY is required")
    return turbopuffer.Turbopuffer(api_key=api_key, region="aws-us-west-2")


def make_voyage_client() -> voyageai.Client:
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY is required")
    return voyageai.Client(api_key=api_key)


def make_openai_client() -> OpenAI:
    api_key = os.environ.get("OAI_KEY")
    if not api_key:
        raise RuntimeError("OAI_KEY is required")
    return OpenAI(api_key=api_key)


# ═══════════════════════════════════════════════════════════════════════
# Multi-pass retrieval
# ═══════════════════════════════════════════════════════════════════════

def retrieve_candidates_multi(
    query: QueryConfig,
    strategy: RetrievalStrategy,
    tpuf_client: turbopuffer.Turbopuffer,
    voyage_client: voyageai.Client,
) -> List[Candidate]:
    """Retrieve candidates using multiple embedding queries and filter sets."""
    ns = tpuf_client.namespace(TPUF_NAMESPACE)
    all_candidates: Dict[str, Candidate] = {}

    # Embed all queries
    embeddings_list = []
    for eq in strategy.embedding_queries:
        resp = voyage_client.embed(eq, model="voyage-3")
        embeddings_list.append(resp.embeddings[0])

    # Build filter variants to try
    filter_variants = []
    if strategy.tpuf_filters_strict is not None:
        filter_variants.append(strategy.tpuf_filters_strict)
    if strategy.tpuf_filters_broad is not None and strategy.tpuf_filters_broad != strategy.tpuf_filters_strict:
        filter_variants.append(strategy.tpuf_filters_broad)
    if not filter_variants:
        filter_variants.append(None)

    for emb in embeddings_list:
        for filt in filter_variants:
            try:
                result = ns.query(
                    rank_by=("vector", "ANN", emb),
                    top_k=strategy.top_k,
                    include_attributes=True,
                    filters=filt,
                )
                for row in (result.rows or []):
                    row_id = str(row.id) if row.id else ""
                    if not row_id:
                        continue
                    # Score is stored as $dist in model_extra
                    row_data = row.model_extra or {}
                    score = float(row_data.get("$dist", 0.0) or 0.0)
                    # All profile attributes are in model_extra
                    data = {k: v for k, v in row_data.items() if k != "$dist"}
                    if row_id in all_candidates:
                        if score > all_candidates[row_id].score:
                            all_candidates[row_id] = Candidate(
                                object_id=row_id, score=score, data=data,
                            )
                    else:
                        all_candidates[row_id] = Candidate(
                            object_id=row_id, score=score, data=data,
                        )
            except Exception as e:
                print(f"  Warning: retrieval pass failed ({e}), continuing...")
                continue

    # Enrich all candidates with parsed degree/experience data
    candidates = list(all_candidates.values())
    for c in candidates:
        enrich_candidate(c)

    print(f"  Retrieved {len(candidates)} unique candidates across all passes")
    return candidates


# ═══════════════════════════════════════════════════════════════════════
# LLM Re-ranking (GPT-4o with structured data)
# ═══════════════════════════════════════════════════════════════════════

def llm_rerank_candidates(
    client: OpenAI,
    query: QueryConfig,
    candidates: List[Candidate],
    max_candidates: int = 50,
) -> List[Candidate]:
    """Re-rank candidates using GPT-4o with full structured profile data."""
    cands = candidates[:max_candidates]
    if not cands:
        return []

    # Build rich profiles
    items = []
    for idx, c in enumerate(cands):
        edu_lines = []
        for deg in c.parsed_degrees:
            parts = []
            if deg.get("degree"):
                parts.append(deg["degree"])
            if deg.get("fos"):
                parts.append(f"in {deg['fos']}")
            if deg.get("school"):
                parts.append(f"from {deg['school']}")
            period = ""
            if deg.get("start") or deg.get("end"):
                period = f" ({deg.get('start', '?')}-{deg.get('end', '') or 'present'})"
            edu_lines.append(" ".join(parts) + period)

        exp_lines = []
        for exp in c.parsed_experiences:
            parts = []
            if exp.get("title"):
                parts.append(exp["title"])
            if exp.get("company"):
                parts.append(f"at {exp['company']}")
            start = exp.get("start", "?")
            end = exp.get("end", "") or "present"
            yrs = exp.get("years", "0")
            parts.append(f"({start}-{end}, ~{yrs}+ yrs)")
            exp_lines.append(" ".join(parts))

        summary = (c.data.get("rerankSummary") or c.data.get("rerank_summary") or "")
        items.append({
            "index": idx,
            "id": c.object_id,
            "name": c.data.get("name", ""),
            "country": c.data.get("country", ""),
            "education": edu_lines,
            "experience": exp_lines,
            "summary": summary[:800],
        })

    system = (
        "You are an expert recruiter evaluating candidates for a specific role.\n\n"
        "EVALUATION RULES:\n"
        "1. HARD CRITERIA are absolute requirements. If a candidate clearly fails ANY hard criterion, score = 0.\n"
        "   - Evaluate EACH hard criterion carefully against education and experience.\n"
        "   - Pay attention to: specific degree types, school quality/location, years of experience, job titles/roles.\n"
        "   - For degree requirements: check the actual degree field, field of study, and school name.\n"
        "   - For experience requirements: check job titles, companies, and duration.\n"
        "2. For candidates passing ALL hard criteria, score 1-10 based on soft criteria:\n"
        "   - 9-10: Exceptional match, strong on all soft criteria\n"
        "   - 7-8: Good match, strong on most soft criteria\n"
        "   - 5-6: Moderate match\n"
        "   - 1-4: Weak soft match but passes hard criteria\n\n"
        "IMPORTANT: Be very strict about hard criteria. If you cannot verify a hard criterion from the data, score 0.\n\n"
        "Return JSON: {\"scores\": [{\"id\": \"...\", \"hard_pass\": true/false, \"score\": N}, ...]}\n"
    )

    all_scores: Dict[str, Tuple[float, bool]] = {}
    batch_size = 20

    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start:batch_start + batch_size]
        user_payload = {
            "role_name": query.name,
            "description": query.description,
            "hard_criteria": query.hard_criteria,
            "soft_criteria": query.soft_criteria,
            "candidates": batch,
        }

        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user_payload)},
                ],
                temperature=0.0,
            )
            content = resp.choices[0].message.content
            if not content:
                continue
            obj = json.loads(content)
            for s in (obj.get("scores") or []):
                cid = str(s.get("id"))
                score = float(s.get("score", 0))
                hard_pass = s.get("hard_pass", True)
                if not hard_pass:
                    score = 0
                all_scores[cid] = (score, hard_pass)
        except Exception as e:
            print(f"  Warning: LLM rerank batch failed: {e}")
            continue

    # Build scored list
    scored = []
    for c in cands:
        if c.object_id in all_scores:
            score_val, _ = all_scores[c.object_id]
            scored.append(Candidate(
                object_id=c.object_id,
                score=score_val,
                data=c.data,
                parsed_degrees=c.parsed_degrees,
                parsed_experiences=c.parsed_experiences,
            ))

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored


# ═══════════════════════════════════════════════════════════════════════
# Evaluation endpoint
# ═══════════════════════════════════════════════════════════════════════

def call_evaluation_endpoint(
    config_path: str,
    object_ids: List[str],
    auth_email: str,
    base_url: str = "https://mercor-dev--search-eng-interview.modal.run/evaluate",
) -> Dict[str, Any]:
    if not object_ids:
        raise ValueError("object_ids must be non-empty.")
    payload = {"config_path": config_path, "object_ids": object_ids}
    headers = {"Content-Type": "application/json", "Authorization": auth_email}
    last_err: Optional[str] = None
    for attempt in range(5):
        try:
            resp = requests.post(base_url, headers=headers, json=payload, timeout=120)
            if resp.status_code >= 500:
                last_err = f"{resp.status_code} {resp.text}"
                time.sleep(2 * (2 ** attempt))
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(2 * (2 ** attempt))
    raise RuntimeError(f"Eval endpoint failed for {config_path}: {last_err}")


# ═══════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline_for_query(
    query: QueryConfig,
    tpuf_client: turbopuffer.Turbopuffer,
    voyage_client: voyageai.Client,
    openai_client: OpenAI,
    auth_email: Optional[str] = None,
    submit: bool = False,
    top_k_submit: int = 10,
) -> Dict[str, Any]:
    """End-to-end pipeline for a single query."""
    print(f"\n{'='*60}")
    print(f"Processing: {query.name} ({query.config_path})")
    print(f"{'='*60}")

    strategy = get_retrieval_strategy(query)

    # Step 1: Multi-pass retrieval
    candidates = retrieve_candidates_multi(query, strategy, tpuf_client, voyage_client)

    # Step 2: Rule-based hard criteria filtering
    if strategy.hard_filter_fn:
        filtered = [c for c in candidates if strategy.hard_filter_fn(c, query)]
        print(f"  After hard filter: {len(filtered)} candidates (from {len(candidates)})")
    else:
        filtered = candidates
        print(f"  No custom hard filter; using all {len(filtered)} candidates")

    # Step 3: Handle edge cases
    if len(filtered) < 10:
        print(f"  Warning: only {len(filtered)} pass hard filter. Supplementing with best ANN matches.")
        filtered_ids = {c.object_id for c in filtered}
        remaining = sorted(
            [c for c in candidates if c.object_id not in filtered_ids],
            key=lambda c: c.score,
            reverse=True,
        )
        # Add more candidates but mark them as supplements
        filtered.extend(remaining[:max(50 - len(filtered), 0)])

    # Step 4: LLM re-ranking with GPT-4o
    # Sort by ANN score first, then take top N for LLM
    filtered.sort(key=lambda c: c.score, reverse=True)
    reranked = llm_rerank_candidates(openai_client, query, filtered[:50])
    print(f"  After LLM rerank: {len(reranked)} scored candidates")

    if reranked:
        ranked_final = reranked
    else:
        ranked_final = sorted(filtered, key=lambda c: c.score, reverse=True)

    top = ranked_final[:top_k_submit]
    top_ids = [c.object_id for c in top]

    print(f"  Top {len(top_ids)} candidates: {[c.data.get('name', '?') for c in top[:5]]}...")

    # Step 5: Evaluate
    eval_result: Optional[Dict[str, Any]] = None
    eval_error: Optional[str] = None
    if submit and auth_email:
        try:
            eval_result = call_evaluation_endpoint(query.config_path, top_ids, auth_email)
            score = eval_result.get("average_final_score", "N/A")
            print(f"  SCORE: {score}")
        except Exception as e:
            eval_error = str(e)
            print(f"  Eval error: {eval_error}")

    return {
        "query_name": query.name,
        "config_path": query.config_path,
        "num_retrieved": len(candidates),
        "num_after_hard_filter": len(filtered),
        "top_ids": top_ids,
        "top_names": [c.data.get("name", "") for c in top],
        "evaluation": eval_result,
        "evaluation_error": eval_error,
    }


def run_all_queries(submit: bool = False) -> List[Dict[str, Any]]:
    tpuf_client = make_tpuf_client()
    voyage_client = make_voyage_client()
    openai_client = make_openai_client()

    auth_email = os.environ.get("MERCOR_EVAL_EMAIL")
    if submit and not auth_email:
        raise RuntimeError("MERCOR_EVAL_EMAIL is required when submit=True.")

    results = []
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

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        score = "N/A"
        if r.get("evaluation"):
            score = r["evaluation"].get("average_final_score", "N/A")
        elif r.get("evaluation_error"):
            score = f"ERROR: {r['evaluation_error'][:60]}"
        print(f"  {r['query_name']:30s} Score: {score}")

    return results


def run_single_query(config_name: str, submit: bool = False) -> Dict[str, Any]:
    """Run pipeline for a single query by config name (e.g. 'tax_lawyer')."""
    tpuf_client = make_tpuf_client()
    voyage_client = make_voyage_client()
    openai_client = make_openai_client()
    auth_email = os.environ.get("MERCOR_EVAL_EMAIL")

    for cfg in get_all_query_configs():
        if cfg.config_path.replace(".yml", "") == config_name:
            return run_pipeline_for_query(
                cfg, tpuf_client, voyage_client, openai_client,
                auth_email=auth_email, submit=submit,
            )
    raise ValueError(f"Unknown config: {config_name}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Improved Mercor search pipeline.")
    parser.add_argument("--submit", action="store_true", help="Submit to evaluation endpoint.")
    parser.add_argument("--output", type=str, default="results.json", help="Output JSON path.")
    parser.add_argument("--query", type=str, default=None, help="Run a single query by config name.")
    args = parser.parse_args()

    if args.query:
        results = [run_single_query(args.query, submit=args.submit)]
    else:
        results = run_all_queries(submit=args.submit)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
