import streamlit as st
import json
import requests
import google.generativeai as genai

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="SR Agent · Search String Builder",
    page_icon="🔍",
    layout="centered",
)

MODEL_NAME = "gemini-2.5-flash"

# ── Styles ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Lora:ital,wght@0,600;1,400&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-title {
    font-family: 'Lora', serif;
    font-size: 1.7rem;
    font-weight: 600;
    color: #0A3D2E;
    margin-bottom: 0.15rem;
}
.main-subtitle {
    font-size: 0.85rem;
    color: #64748B;
    margin-bottom: 1.5rem;
}
.step-badge {
    display: inline-block;
    background: #E8F5EE;
    color: #0D5C4E;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    letter-spacing: 0.03em;
    margin-bottom: 0.5rem;
}
.pico-mini {
    background: #F8FDFB;
    border: 1px solid #BBD8C8;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 1.5rem;
    font-size: 0.82rem;
    color: #1E293B;
    line-height: 1.8;
}
.pico-mini strong {
    color: #0D5C4E;
    display: inline-block;
    width: 130px;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.db-note {
    font-size: 0.78rem;
    color: #64748B;
    margin-bottom: 0.5rem;
    line-height: 1.5;
}
.manual-badge {
    display: inline-block;
    background: #FFF7ED;
    color: #9A3412;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    margin-bottom: 0.5rem;
}
.auto-badge {
    display: inline-block;
    background: #F0FDF7;
    color: #0D5C4E;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── API key resolution ───────────────────────────────────────
def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return st.session_state.get("api_key_input", "")

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 SR Agent")
    st.caption("AI-assisted systematic review pipeline")
    st.divider()

    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIza…",
            key="api_key_input",
        )

    st.divider()
    steps = [
        ("1", "PICO Refiner",          False),
        ("2", "Search String Builder",  True),
        ("3", "Abstract Screener",      False),
        ("4", "Quality Appraisal",      False),
        ("5", "Data Extraction",        False),
    ]
    st.markdown("**Pipeline**")
    for num, name, active in steps:
        if active:
            st.markdown(f"**→ Step {num}: {name}**")
        else:
            st.markdown(
                f"<span style='color:#94A3B8'>Step {num}: {name}</span>",
                unsafe_allow_html=True,
            )

# ── Guard: need PICO from Step 1 ────────────────────────────
if "pico" not in st.session_state or not st.session_state.pico:
    st.warning("No confirmed PICO found. Please complete Step 1 first.")
    if st.button("← Go to Step 1"):
        st.switch_page("app.py")
    st.stop()

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to continue.")
    st.stop()

genai.configure(api_key=api_key)

pico = st.session_state.pico

# ── Header ───────────────────────────────────────────────────
st.markdown('<div class="step-badge">Step 2 of 5</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Search String Builder</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Generating optimised Boolean search strings '
    'for all four databases from your confirmed PICO.</div>',
    unsafe_allow_html=True,
)

# ── Compact PICO summary ─────────────────────────────────────
rows = [
    ("P — Population",   pico.get("P", "—")),
    ("I — Intervention", pico.get("I", "—")),
    ("C — Comparator",   pico.get("C", "—")),
    ("O — Outcome",      pico.get("O", "—")),
]
if pico.get("study_design"):
    rows.append(("Study Design", pico["study_design"]))

rows_html = "".join(
    f"<div><strong>{label}</strong> {value}</div>"
    for label, value in rows
)
st.markdown(f'<div class="pico-mini">{rows_html}</div>', unsafe_allow_html=True)

# ── Generate strings (once, cached in session state) ─────────
SEARCH_PROMPT = """
You are a systematic review librarian expert at constructing Boolean search strings.

Given this confirmed PICO:
P (Population):   {P}
I (Intervention): {I}
C (Comparator):   {C}
O (Outcome):      {O}
Study design:     {study_design}

Generate optimised Boolean search strings for FOUR databases.
For each concept group, include synonyms, spelling variants, and abbreviations combined with OR.
Combine concept groups with AND.

Return ONLY a valid JSON object in exactly this format — no explanation, no markdown:
{{
  "pubmed": "full PubMed string using ([MeSH Terms] OR [tiab]) pattern",
  "openalex": "simplified free-text query with key synonyms, no field codes",
  "scopus": "Scopus string using TITLE-ABS-KEY() with AND/OR",
  "wos": "Web of Science string using TS=() syntax"
}}

Rules:
- PubMed: use ([MeSH Terms] OR \"term\"[tiab]) pattern per concept; join concepts with AND
- Scopus: TITLE-ABS-KEY(concept1 AND concept2); use W/n proximity where helpful
- WOS: TS=(concept1 AND concept2); use SAME/n proximity where helpful
- OpenAlex: plain keyword string, no field tags, most discriminating terms only
- All strings must be ready to paste directly into each database with no editing needed
- Include relevant MeSH terms for PubMed even if not in the PICO — use clinical knowledge
""".strip()

if "search_strings" not in st.session_state:
    st.session_state.search_strings = None

if st.session_state.search_strings is None:
    with st.spinner("Generating search strings for all four databases…"):
        prompt = SEARCH_PROMPT.format(
            P=pico.get("P", ""),
            I=pico.get("I", ""),
            C=pico.get("C", ""),
            O=pico.get("O", ""),
            study_design=pico.get("study_design", "Not specified"),
        )
        model = genai.GenerativeModel(model_name=MODEL_NAME)
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            st.session_state.search_strings = json.loads(raw)
        except json.JSONDecodeError:
            st.error("Could not parse Gemini response. Please try regenerating.")
            if st.button("🔄 Retry"):
                st.rerun()
            st.stop()

strings = st.session_state.search_strings

# ── Refine button ────────────────────────────────────────────
col_head, col_btn = st.columns([3, 1])
with col_head:
    st.markdown("### Search strings")
with col_btn:
    if st.button("🔄 Regenerate", use_container_width=True):
        st.session_state.search_strings = None
        st.rerun()

# ── Database tabs ────────────────────────────────────────────
tab_pm, tab_oa, tab_sc, tab_wos = st.tabs(["PubMed", "OpenAlex", "Scopus", "Web of Science"])

# ── PubMed ───────────────────────────────────────────────────
with tab_pm:
    st.markdown('<span class="auto-badge">⚡ Automated in pipeline</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="db-note">MeSH terms + free-text [tiab] pattern. '
        'Run automatically by the pipeline — no copy-paste needed.</div>',
        unsafe_allow_html=True,
    )
    st.code(strings.get("pubmed", ""), language="text")

    if st.button("▶ Test on PubMed now", key="test_pm"):
        with st.spinner("Querying PubMed…"):
            try:
                r = requests.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                    params={
                        "db": "pubmed",
                        "term": strings["pubmed"],
                        "retmode": "json",
                        "retmax": 0,
                    },
                    timeout=15,
                )
                count = int(r.json()["esearchresult"]["count"])
                if count == 0:
                    st.error(f"0 results — string may be too specific. Try Regenerate.")
                elif count < 20:
                    st.warning(f"**{count:,} results** — quite narrow. Consider broadening.")
                elif count <= 500:
                    st.success(f"**{count:,} results** — good scope. ✓")
                elif count <= 2000:
                    st.warning(f"**{count:,} results** — broad. Consider tightening.")
                else:
                    st.error(f"**{count:,} results** — too broad. Narrow the string.")
            except Exception as e:
                st.error(f"PubMed error: {e}")

# ── OpenAlex ─────────────────────────────────────────────────
with tab_oa:
    st.markdown('<span class="auto-badge">⚡ Automated in pipeline</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="db-note">Simplified free-text query. '
        'Run automatically by the pipeline — no copy-paste needed.</div>',
        unsafe_allow_html=True,
    )
    st.code(strings.get("openalex", ""), language="text")

    if st.button("▶ Test on OpenAlex now", key="test_oa"):
        with st.spinner("Querying OpenAlex…"):
            try:
                r = requests.get(
                    "https://api.openalex.org/works",
                    params={"search": strings["openalex"], "per-page": 1},
                    headers={"User-Agent": "SR-Agent/1.0 (mailto:sr-agent@ukm.edu.my)"},
                    timeout=15,
                )
                count = r.json().get("meta", {}).get("count", 0)
                if count == 0:
                    st.error(f"0 results — string may be too specific.")
                elif count < 20:
                    st.warning(f"**{count:,} results** — quite narrow.")
                elif count <= 1000:
                    st.success(f"**{count:,} results** — good scope. ✓")
                elif count <= 5000:
                    st.warning(f"**{count:,} results** — broad.")
                else:
                    st.error(f"**{count:,} results** — too broad.")
            except Exception as e:
                st.error(f"OpenAlex error: {e}")

# ── Scopus ───────────────────────────────────────────────────
with tab_sc:
    st.markdown('<span class="manual-badge">✋ Manual — copy and paste</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="db-note">Copy this string and paste it into the '
        '<a href="https://www.scopus.com/search/form.uri" target="_blank">Scopus Advanced Search</a>. '
        'Export results as CSV or RIS, then upload in Step 3.</div>',
        unsafe_allow_html=True,
    )
    st.code(strings.get("scopus", ""), language="text")

# ── Web of Science ───────────────────────────────────────────
with tab_wos:
    st.markdown('<span class="manual-badge">✋ Manual — copy and paste</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="db-note">Copy this string and paste it into the '
        '<a href="https://www.webofscience.com/wos/woscc/advanced-search" target="_blank">WOS Advanced Search</a>. '
        'Export results as plain text or RIS, then upload in Step 3.</div>',
        unsafe_allow_html=True,
    )
    st.code(strings.get("wos", ""), language="text")

# ── Ask Gemini to refine a specific string ───────────────────
st.divider()
st.markdown("#### Refine a string")
st.caption("Ask Gemini to adjust a specific string — e.g. 'Add more MeSH terms to PubMed' or 'Narrow the Scopus string'.")

refine_input = st.text_input("Your instruction", placeholder="e.g. The PubMed string is too broad — remove the outcome terms")
if st.button("Refine", disabled=not refine_input):
    with st.spinner("Refining…"):
        refine_prompt = f"""
You previously generated these search strings for a systematic review:

PubMed:    {strings.get('pubmed', '')}
OpenAlex:  {strings.get('openalex', '')}
Scopus:    {strings.get('scopus', '')}
WOS:       {strings.get('wos', '')}

The researcher's instruction: {refine_input}

Apply the instruction and return ONLY an updated JSON object in the same format:
{{"pubmed": "...", "openalex": "...", "scopus": "...", "wos": "..."}}
Only change the strings affected by the instruction — keep others identical.
""".strip()
        model = genai.GenerativeModel(model_name=MODEL_NAME)
        response = model.generate_content(refine_prompt)
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            st.session_state.search_strings = json.loads(raw.strip())
            st.rerun()
        except Exception:
            st.error("Could not parse refined strings. Try rephrasing your instruction.")

# ── Continue ─────────────────────────────────────────────────
st.divider()
st.info("✅ Step 2 complete — once you have your Scopus and WOS exports ready, continue to Step 3: Abstract Screener.")
