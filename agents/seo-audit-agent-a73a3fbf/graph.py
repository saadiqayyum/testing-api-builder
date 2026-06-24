"""
SEO Audit Agent
---------------
Takes a URL, audits on-page SEO signals, and returns a prioritized
list of fixes (each with a severity level) powered by an LLM.

Nodes
-----
1. fetch_page       – HTTP GET + BeautifulSoup parse
2. audit_seo        – Rule-based checks for every on-page SEO element
3. prioritize_fixes – LLM ranks the findings and writes actionable advice
"""

import os
import json
import re
import textwrap
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SEOState(TypedDict):
    # Input
    url: str

    # Populated by fetch_page
    html: Optional[str]
    fetch_error: Optional[str]
    status_code: Optional[int]
    final_url: Optional[str]          # after redirects

    # Populated by audit_seo
    raw_findings: Optional[list[dict]]   # list of raw issue dicts
    page_snapshot: Optional[dict]        # summary of what was found

    # Populated by prioritize_fixes
    fixes: Optional[list[dict]]          # final prioritized output
    summary: Optional[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; SEO-Audit-Bot/1.0; +https://orkest.ai)"
    )
}

def _text_len(text: str) -> int:
    return len(text.strip())


def _count_words(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# Node 1 – fetch_page
# ---------------------------------------------------------------------------

def fetch_page(state: SEOState) -> dict[str, Any]:
    url = state["url"].strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        return {
            "html": resp.text,
            "status_code": resp.status_code,
            "final_url": resp.url,
            "fetch_error": None,
        }
    except requests.RequestException as exc:
        return {
            "html": None,
            "status_code": None,
            "final_url": url,
            "fetch_error": str(exc),
        }


# ---------------------------------------------------------------------------
# Node 2 – audit_seo
# ---------------------------------------------------------------------------

def audit_seo(state: SEOState) -> dict[str, Any]:
    if state.get("fetch_error") or not state.get("html"):
        return {
            "raw_findings": [
                {
                    "element": "Page Fetch",
                    "issue": f"Could not retrieve page: {state.get('fetch_error', 'unknown error')}",
                    "current_value": None,
                }
            ],
            "page_snapshot": {},
        }

    soup = BeautifulSoup(state["html"], "html.parser")
    findings: list[dict] = []
    snapshot: dict[str, Any] = {}

    # ── Title ────────────────────────────────────────────────────────────────
    title_tag = soup.find("title")
    title_text = title_tag.get_text(strip=True) if title_tag else ""
    title_len = _text_len(title_text)
    snapshot["title"] = title_text

    if not title_text:
        findings.append({
            "element": "Title Tag",
            "issue": "Title tag is missing entirely.",
            "current_value": None,
            "recommendation": "Add a descriptive <title> tag (50–60 characters).",
        })
    elif title_len < 30:
        findings.append({
            "element": "Title Tag",
            "issue": f"Title is too short ({title_len} chars). Short titles miss keyword opportunities.",
            "current_value": title_text,
            "recommendation": "Expand the title to 50–60 characters with primary keywords.",
        })
    elif title_len > 60:
        findings.append({
            "element": "Title Tag",
            "issue": f"Title is too long ({title_len} chars) and may be truncated in SERPs.",
            "current_value": title_text,
            "recommendation": "Trim the title to under 60 characters while keeping keywords.",
        })

    # ── Meta Description ─────────────────────────────────────────────────────
    meta_desc_tag = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    meta_desc = meta_desc_tag.get("content", "").strip() if meta_desc_tag else ""
    meta_desc_len = _text_len(meta_desc)
    snapshot["meta_description"] = meta_desc

    if not meta_desc:
        findings.append({
            "element": "Meta Description",
            "issue": "Meta description is missing.",
            "current_value": None,
            "recommendation": "Add a meta description of 120–158 characters summarising the page.",
        })
    elif meta_desc_len < 70:
        findings.append({
            "element": "Meta Description",
            "issue": f"Meta description is too short ({meta_desc_len} chars).",
            "current_value": meta_desc,
            "recommendation": "Expand to 120–158 characters with a natural call-to-action.",
        })
    elif meta_desc_len > 158:
        findings.append({
            "element": "Meta Description",
            "issue": f"Meta description is too long ({meta_desc_len} chars) and will be truncated.",
            "current_value": meta_desc[:200],
            "recommendation": "Shorten to 120–158 characters.",
        })

    # ── Canonical ────────────────────────────────────────────────────────────
    canonical_tag = soup.find("link", attrs={"rel": re.compile(r"canonical", re.I)})
    canonical_href = canonical_tag.get("href", "").strip() if canonical_tag else ""
    snapshot["canonical"] = canonical_href

    if not canonical_href:
        findings.append({
            "element": "Canonical Tag",
            "issue": "No canonical tag found. Duplicate content risk.",
            "current_value": None,
            "recommendation": f'Add <link rel="canonical" href="{state["final_url"]}" /> to the <head>.',
        })
    else:
        # Warn if canonical points elsewhere
        parsed_input = urlparse(state.get("final_url", state["url"]))
        parsed_canon = urlparse(canonical_href)
        if parsed_canon.netloc and parsed_canon.netloc != parsed_input.netloc:
            findings.append({
                "element": "Canonical Tag",
                "issue": "Canonical points to a different domain — verify this is intentional.",
                "current_value": canonical_href,
                "recommendation": "Ensure canonical references the preferred version of this page.",
            })

    # ── Headings ─────────────────────────────────────────────────────────────
    h1_tags = soup.find_all("h1")
    h2_tags = soup.find_all("h2")
    h3_tags = soup.find_all("h3")
    snapshot["h1_count"] = len(h1_tags)
    snapshot["h2_count"] = len(h2_tags)
    snapshot["h1_texts"] = [h.get_text(strip=True) for h in h1_tags]

    if len(h1_tags) == 0:
        findings.append({
            "element": "H1 Heading",
            "issue": "No H1 heading found on the page.",
            "current_value": None,
            "recommendation": "Add exactly one H1 that includes the primary keyword.",
        })
    elif len(h1_tags) > 1:
        findings.append({
            "element": "H1 Heading",
            "issue": f"Multiple H1 tags found ({len(h1_tags)}). This dilutes keyword signals.",
            "current_value": [h.get_text(strip=True) for h in h1_tags],
            "recommendation": "Keep exactly one H1 per page; demote extras to H2/H3.",
        })

    if len(h2_tags) == 0 and len(soup.get_text(strip=True)) > 300:
        findings.append({
            "element": "H2 Headings",
            "issue": "No H2 headings found. Long content lacks structural hierarchy.",
            "current_value": None,
            "recommendation": "Break content into sections with descriptive H2 headings.",
        })

    # Check for heading hierarchy skips (e.g. H1 → H3 without H2)
    all_headings = soup.find_all(re.compile(r"^h[1-6]$"))
    heading_levels = [int(h.name[1]) for h in all_headings]
    for i in range(1, len(heading_levels)):
        if heading_levels[i] - heading_levels[i - 1] > 1:
            findings.append({
                "element": "Heading Hierarchy",
                "issue": (
                    f"Heading level skips from H{heading_levels[i-1]} to "
                    f"H{heading_levels[i]} — breaks semantic outline."
                ),
                "current_value": heading_levels,
                "recommendation": "Ensure headings increment by one level at a time (H1 → H2 → H3).",
            })
            break  # report once

    # ── Images / Alt Text ────────────────────────────────────────────────────
    images = soup.find_all("img")
    missing_alt = [img for img in images if not img.get("alt")]
    empty_alt   = [img for img in images if img.get("alt") is not None and img.get("alt").strip() == ""]
    snapshot["total_images"] = len(images)
    snapshot["images_missing_alt"] = len(missing_alt)
    snapshot["images_empty_alt"] = len(empty_alt)

    if missing_alt:
        sample_srcs = [img.get("src", "")[:80] for img in missing_alt[:5]]
        findings.append({
            "element": "Image Alt Text",
            "issue": (
                f"{len(missing_alt)} image(s) have no alt attribute at all "
                f"(out of {len(images)} total)."
            ),
            "current_value": sample_srcs,
            "recommendation": (
                "Add descriptive alt attributes to every content image; "
                "use alt=\"\" only for decorative images."
            ),
        })
    if empty_alt:
        findings.append({
            "element": "Image Alt Text",
            "issue": f"{len(empty_alt)} image(s) have an empty alt attribute.",
            "current_value": None,
            "recommendation": (
                "Replace empty alt attributes with descriptive text "
                "if the images convey meaningful content."
            ),
        })

    # ── Open Graph / Social Tags ─────────────────────────────────────────────
    og_title = soup.find("meta", property="og:title")
    og_desc  = soup.find("meta", property="og:description")
    og_image = soup.find("meta", property="og:image")
    snapshot["has_og_title"] = bool(og_title)
    snapshot["has_og_description"] = bool(og_desc)
    snapshot["has_og_image"] = bool(og_image)

    missing_og = []
    if not og_title:   missing_og.append("og:title")
    if not og_desc:    missing_og.append("og:description")
    if not og_image:   missing_og.append("og:image")
    if missing_og:
        findings.append({
            "element": "Open Graph Tags",
            "issue": f"Missing Open Graph tags: {', '.join(missing_og)}.",
            "current_value": None,
            "recommendation": (
                "Add og:title, og:description, and og:image to improve "
                "appearance when shared on social platforms."
            ),
        })

    # ── Robots Meta ──────────────────────────────────────────────────────────
    robots_meta = soup.find("meta", attrs={"name": re.compile(r"^robots$", re.I)})
    robots_content = robots_meta.get("content", "").lower() if robots_meta else ""
    snapshot["robots_meta"] = robots_content

    if "noindex" in robots_content:
        findings.append({
            "element": "Robots Meta",
            "issue": "Page has meta robots=noindex — it will not be indexed by search engines.",
            "current_value": robots_content,
            "recommendation": "Remove noindex if this page should appear in search results.",
        })

    # ── Page Word Count ──────────────────────────────────────────────────────
    body = soup.find("body")
    body_text = body.get_text(" ", strip=True) if body else ""
    word_count = _count_words(body_text)
    snapshot["word_count"] = word_count

    if word_count < 300:
        findings.append({
            "element": "Content Depth",
            "issue": f"Page has only ~{word_count} words. Thin content ranks poorly.",
            "current_value": word_count,
            "recommendation": "Expand content to at least 600 words covering the topic thoroughly.",
        })

    # ── Lang Attribute ───────────────────────────────────────────────────────
    html_tag = soup.find("html")
    lang_attr = html_tag.get("lang", "").strip() if html_tag else ""
    snapshot["lang"] = lang_attr
    if not lang_attr:
        findings.append({
            "element": "HTML Lang Attribute",
            "issue": "The <html> tag is missing a lang attribute.",
            "current_value": None,
            "recommendation": 'Add lang="en" (or appropriate language code) to the <html> tag.',
        })

    return {"raw_findings": findings, "page_snapshot": snapshot}


# ---------------------------------------------------------------------------
# Node 3 – prioritize_fixes
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a senior SEO consultant. You receive a list of raw audit findings
    for a webpage and must return a prioritized action plan.

    For EACH finding, output:
    - severity: one of  CRITICAL | HIGH | MEDIUM | LOW
    - element:  the SEO element (e.g. "Title Tag")
    - issue:    concise description of the problem
    - fix:      the specific, actionable fix (1–3 sentences)
    - why:      why this matters for rankings / user experience (1 sentence)

    Severity guide:
      CRITICAL – directly blocks indexing or massively hurts rankings
      HIGH     – significant ranking or click-through impact
      MEDIUM   – meaningful improvement opportunity
      LOW      – best-practice polish

    Sort the list from CRITICAL down to LOW (ties: keep original order).
    Return ONLY a JSON array of objects with keys:
      severity, element, issue, fix, why

    If there are no findings, return an empty array [].
""")


def prioritize_fixes(state: SEOState) -> dict[str, Any]:
    findings = state.get("raw_findings", [])
    snapshot = state.get("page_snapshot", {})
    url = state.get("final_url") or state.get("url")

    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )

    user_content = json.dumps(
        {
            "url": url,
            "page_snapshot": snapshot,
            "raw_findings": findings,
        },
        indent=2,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        fixes = json.loads(raw)
    except json.JSONDecodeError:
        fixes = [
            {
                "severity": "HIGH",
                "element": "Audit",
                "issue": "Could not parse LLM response as JSON.",
                "fix": raw[:500],
                "why": "Manual review required.",
            }
        ]

    # Build a human-readable summary
    if fixes:
        counts = {}
        for f in fixes:
            s = f.get("severity", "UNKNOWN")
            counts[s] = counts.get(s, 0) + 1
        count_str = ", ".join(f"{v} {k}" for k, v in counts.items())
        summary = (
            f"Found {len(fixes)} issue(s) for {url} — {count_str}. "
            "See `fixes` for the full prioritized action plan."
        )
    else:
        summary = f"No SEO issues found for {url}. The page looks well-optimised!"

    return {"fixes": fixes, "summary": summary}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(SEOState)

    builder.add_node("fetch_page", fetch_page)
    builder.add_node("audit_seo", audit_seo)
    builder.add_node("prioritize_fixes", prioritize_fixes)

    builder.add_edge(START, "fetch_page")
    builder.add_edge("fetch_page", "audit_seo")
    builder.add_edge("audit_seo", "prioritize_fixes")
    builder.add_edge("prioritize_fixes", END)

    return builder.compile()


graph = build_graph()
