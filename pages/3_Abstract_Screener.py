import streamlit as st
import json
import time
import csv
import io
import requests
import xml.etree.ElementTree as ET
import google.generativeai as genai

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="SR Agent · Abstract Screener",
    page_icon="📋",
    layout="centered",
)

MODEL_NAME = "gemini-2.5-flash"

# ── Styles ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Lora:ital,wght@0,600;1,400&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-title { font-family:'Lora',serif; font-size:1.7rem; font-weight:600; color:#0A3D2E; margin-bottom:0.15rem; }
.main-subtitle { font-size:0.85rem; color:#64748B; margin-bottom:1.5rem; }
.step-badge { display:inline-block; background:#E8F5EE; color:#0D5C4E; font-size:0.7rem; font-weight:600; padding:0.2rem 0.6rem; border-radius:20px; letter-spacing:0.03em; margin-bottom:0.5rem; }
.pico-mini { background:#F8FDFB; border:1px solid #BBD8C8; border-radius:8px; padding:0.9rem 1.2rem; margin-bottom:1.5rem; font-size:0.82rem; color:#1E293B; line-height:1.8; }
.pico-mini strong { color:#0D5C4E; display:inline-block; width:130px; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.05em; }
.paper-card { background:#fff; border:1.5px solid #E2E8F0; border-radius:10px; padding:1.25rem 1.5rem; margin-bottom:1rem; }
.paper-title { font-family:'Lora',serif; font-size:1rem; font-weight:600; color:#1A1F36; margin-bottom:0.3rem; line-height:1.45; }
.paper-meta { font-size:0.75rem; color:#64748B; margin-bottom:0.75rem; }
.paper-abstract { font-size:0.83rem; color:#334155; line-height:1.65; max-height:180px; overflow-y:auto; margin-bottom:0.75rem; }
.ai-include { background:#F0FDF7; color:#0D5C4E; border:1px solid #BBD8C8; border-radius:6px; padding:0.4rem 0.75rem; font-size:0.78rem; font-weight:600; display:inline-block; margin-bottom:0.5rem; }
.ai-exclude { background:#FEF2F2; color:#B91C1C; border:1px solid #FECACA; border-radius:6px; padding:0.4rem 0.75rem; font-size:0.78rem; font-weight:600; display:inline-block; margin-bottom:0.5rem; }
.ai-uncertain { background:#FFFBEB; color:#92400E; border:1px solid #FDE68A; border-radius:6px; padding:0.4rem 0.75rem; font-size:0.78rem; font-weight:600; display:inline-block; margin-bottom:0.5rem; }
.ai-reason { font-size:0.78rem; color:#64748B; font-style:italic; margin-bottom:0.75rem; }
.src-badge { display:inline-block; background:#F1F5F9; color:#475569; font-size:0.65rem; font-weight:600; padding:0.1rem 0.45rem; border-radius:4px; margin-right:0.3rem; }
</style>
""", unsafe_allow_html=True)

# ── API key ──────────────────────────────────────────────────
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
        api_key = st.text_input("Gemini API Key", type="password", placeholder="AIza…", key="api_key_input")
    st.divider()
    steps = [("1","PICO Refiner",False),("2","Search String Builder",False),("3","Abstract Screener",True),("4","Quality Appraisal",False),("5","Data Extraction",False)]
    st.markdown("**Pipeline**")
    for num, name, active in steps:
        if active:
            st.markdown(f"**→ Step {num}: {name}**")
        else:
            st.markdown(f"<span style='color:#94A3B8'>Step {num}: {name}</span>", unsafe_allow_html=True)
    st.divider()
    if st.button("↺ Reset screener", use_container_width=True):
        for k in ["papers_raw","papers_deduped","screening_done","screening_results","review_index"]:
            st.session_state.pop(k, None)
        st.rerun()

# ── Guards ───────────────────────────────────────────────────
if "pico" not in st.session_state or not st.session_state.pico:
    st.warning("No confirmed PICO found. Please complete Step 1 first.")
    if st.button("← Go to Step 1"):
        st.switch_page("app.py")
    st.stop()

if "search_strings" not in st.session_state or not st.session_state.search_strings:
    st.warning("No search strings found. Please complete Step 2 first.")
    if st.button("← Go to Step 2"):
        st.switch_page("pages/2_Search_String_Builder.py")
    st.stop()

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to continue.")
    st.stop()

genai.configure(api_key=api_key)
pico    = st.session_state.pico
strings = st.session_state.search_strings

# ── Session state ────────────────────────────────────────────
for k, v in {
    "papers_raw":        None,
    "papers_deduped":    None,
    "screening_done":    False,
    "screening_results": {},
    "review_index":      0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Paper parsers ────────────────────────────────────────────
def reconstruct_abstract(inv):
    if not inv:
        return ""
    pos = {}
    for word, positions in inv.items():
        for p in positions:
            pos[p] = word
    return " ".join(pos[i] for i in sorted(pos))

def fetch_pubmed(search_string, max_results=300):
    try:
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db":"pubmed","term":search_string,"retmode":"json","retmax":max_results},
            timeout=15,
        )
        pmids = r.json()["esearchresult"]["idlist"]
        if not pmids:
            return []
        papers = []
        for i in range(0, len(pmids), 100):
            batch = pmids[i:i+100]
            r2 = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db":"pubmed","id":",".join(batch),"retmode":"xml"},
                timeout=30,
            )
            root = ET.fromstring(r2.content)
            for art in root.findall(".//PubmedArticle"):
                title    = art.findtext(".//ArticleTitle") or ""
                abstract = " ".join(e.text or "" for e in art.findall(".//AbstractText"))
                doi      = ""
                for eid in art.findall(".//ArticleId"):
                    if eid.get("IdType") == "doi":
                        doi = eid.text or ""
                year   = art.findtext(".//PubDate/Year") or ""
                jrnl   = art.findtext(".//Journal/Title") or ""
                aus    = [f"{a.findtext('LastName','')} {a.findtext('Initials','')}" for a in art.findall(".//Author")]
                papers.append({"title":title.strip(),"abstract":abstract.strip(),"doi":doi.lower().strip(),"year":year,"journal":jrnl,"authors":", ".join(aus[:3]),"source":"PubMed"})
        return papers
    except Exception as e:
        st.error(f"PubMed fetch error: {e}")
        return []

def fetch_openalex(search_string, max_results=300):
    papers = []
    page   = 1
    per_pg = 50
    try:
        while len(papers) < max_results:
            r = requests.get(
                "https://api.openalex.org/works",
                params={"search":search_string,"per-page":per_pg,"page":page,"select":"title,abstract_inverted_index,doi,publication_year,primary_location,authorships"},
                headers={"User-Agent":"SR-Agent/1.0 (mailto:sr-agent@ukm.edu.my)"},
                timeout=15,
            )
            works = r.json().get("results",[])
            if not works:
                break
            for w in works:
                doi  = (w.get("doi") or "").replace("https://doi.org/","").lower().strip()
                jrnl = ""
                if w.get("primary_location") and w["primary_location"].get("source"):
                    jrnl = w["primary_location"]["source"].get("display_name","")
                aus  = [a["author"]["display_name"] for a in (w.get("authorships") or [])[:3]]
                papers.append({
                    "title":     (w.get("title") or "").strip(),
                    "abstract":  reconstruct_abstract(w.get("abstract_inverted_index")),
                    "doi":       doi,
                    "year":      str(w.get("publication_year") or ""),
                    "journal":   jrnl,
                    "authors":   ", ".join(aus),
                    "source":    "OpenAlex",
                })
            page += 1
            if len(works) < per_pg:
                break
    except Exception as e:
        st.error(f"OpenAlex fetch error: {e}")
    return papers[:max_results]

def parse_ris(content, source_label="Upload"):
    papers, current, cur_tag = [], {}, None
    for line in content.splitlines():
        line = line.rstrip()
        if line.startswith("ER"):
            if current.get("title"):
                current.setdefault("abstract",""); current.setdefault("doi","")
                current.setdefault("year","");    current.setdefault("journal","")
                current["authors"] = ", ".join(current.pop("_aus",[])[:3])
                current["source"]  = source_label
                papers.append(current)
            current, cur_tag = {}, None
        elif "  - " in line:
            tag, val = line.split("  - ", 1)
            tag = tag.strip()
            if tag == "TI":   current["title"]   = val; cur_tag = "TI"
            elif tag == "AB": current["abstract"] = val; cur_tag = "AB"
            elif tag == "DO": current["doi"]      = val.lower().strip(); cur_tag = None
            elif tag in ("PY","Y1"): current["year"] = val[:4]; cur_tag = None
            elif tag in ("JO","T2","SO","JF"): current.setdefault("journal", val); cur_tag = None
            elif tag == "AU": current.setdefault("_aus",[]).append(val); cur_tag = "AU"
            else: cur_tag = None
        elif line.startswith("  ") and cur_tag == "AB":
            current["abstract"] = current.get("abstract","") + " " + line.strip()
    return papers

def parse_scopus_csv(content):
    papers = []
    try:
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            title = row.get("Title", row.get("title","")).strip()
            if not title:
                continue
            papers.append({
                "title":    title,
                "abstract": row.get("Abstract", row.get("abstract","")).strip(),
                "doi":      row.get("DOI", row.get("doi","")).lower().strip(),
                "year":     str(row.get("Year", row.get("year",""))).strip(),
                "journal":  row.get("Source title", row.get("source title","")).strip(),
                "authors":  row.get("Authors", row.get("authors","")).strip(),
                "source":   "Scopus",
            })
    except Exception as e:
        st.error(f"Scopus CSV parse error: {e}")
    return papers

def parse_wos(content):
    papers, current, cur_tag = [], {}, None
    for line in content.splitlines():
        if len(line) >= 2:
            tag = line[:2]
            val = line[3:].strip() if len(line) > 3 else ""
            if tag == "ER":
                if current.get("title"):
                    current.setdefault("abstract",""); current.setdefault("doi","")
                    current.setdefault("year","");    current.setdefault("journal","")
                    current["authors"] = ", ".join(current.pop("_aus",[])[:3])
                    current["source"]  = "WOS"
                    papers.append(current)
                current, cur_tag = {}, None
            elif tag == "TI": current["title"]   = val; cur_tag = "TI"
            elif tag == "AB": current["abstract"] = val; cur_tag = "AB"
            elif tag == "DO": current["doi"]      = val.lower().strip(); cur_tag = None
            elif tag == "PY": current["year"]     = val; cur_tag = None
            elif tag == "SO": current["journal"]  = val; cur_tag = None
            elif tag == "AU": current.setdefault("_aus",[]).append(val); cur_tag = "AU"
            elif line.startswith("   ") and cur_tag == "AB":
                current["abstract"] = current.get("abstract","") + " " + line.strip()
    return papers

def deduplicate(papers):
    seen_doi, seen_title, unique = set(), {}, []
    for p in papers:
        doi   = p.get("doi","").strip()
        title = p.get("title","").lower().strip()
        if doi and doi in seen_doi:
            continue
        is_dup = any(
            title and t and len(title) > 15 and (title == t or title in t or t in title)
            for t in seen_title
        )
        if not is_dup:
            unique.append(p)
            if doi:
                seen_doi.add(doi)
            if title:
                seen_title[title] = True
    return unique

# ── AI batch screener ────────────────────────────────────────
SCREEN_PROMPT = """You are screening abstracts for a systematic review.

Research question: {fq}
PICO — P: {P} | I: {I} | C: {C} | O: {O}

Screen each paper below. Decide INCLUDE, EXCLUDE, or UNCERTAIN.
- INCLUDE: appears to meet PICO criteria based on title/abstract
- EXCLUDE: clearly does not meet criteria
- UNCERTAIN: cannot determine from title/abstract alone

Papers (JSON array):
{papers}

Return ONLY a JSON array — no markdown, no explanation:
[{{"id":0,"decision":"INCLUDE","reason":"one sentence"}}, ...]
One entry per paper, same order as input."""

def screen_batch(batch, pico, model):
    papers_json = json.dumps([
        {"id": i, "title": p.get("title",""), "abstract": p.get("abstract","")[:600]}
        for i, p in enumerate(batch)
    ])
    prompt = SCREEN_PROMPT.format(
        fq=pico.get("formal_question",""),
        P=pico.get("P",""), I=pico.get("I",""),
        C=pico.get("C",""), O=pico.get("O",""),
        papers=papers_json,
    )
    try:
        resp = model.generate_content(prompt)
        raw  = resp.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        return json.loads(raw)
    except Exception as e:
        return [{"id":i,"decision":"UNCERTAIN","reason":f"Screening error: {e}"} for i in range(len(batch))]

# ── Header ───────────────────────────────────────────────────
st.markdown('<div class="step-badge">Step 3 of 5</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Abstract Screener</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Collect papers from all sources, run AI screening, then confirm each decision.</div>', unsafe_allow_html=True)

# ── Compact PICO ─────────────────────────────────────────────
rows_html = "".join(f"<div><strong>{k}</strong> {v}</div>" for k,v in [
    ("P — Population", pico.get("P","—")), ("I — Intervention", pico.get("I","—")),
    ("C — Comparator",  pico.get("C","—")), ("O — Outcome",       pico.get("O","—")),
])
st.markdown(f'<div class="pico-mini">{rows_html}</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PHASE 1 — Paper collection
# ════════════════════════════════════════════════════════════
if st.session_state.papers_deduped is None:
    st.markdown("### 1 · Collect papers")

    # ── Auto-fetch ───────────────────────────────────────────
    col1, col2 = st.columns(2)
    pm_papers, oa_papers = [], []

    with col1:
        st.markdown("**⚡ Auto-fetch**")
        if st.button("Fetch PubMed", use_container_width=True):
            with st.spinner("Fetching from PubMed…"):
                pm_papers = fetch_pubmed(strings.get("pubmed",""))
            st.session_state["_pm"] = pm_papers
            st.success(f"{len(pm_papers)} papers from PubMed")
        if "_pm" in st.session_state:
            st.caption(f"PubMed: {len(st.session_state['_pm'])} papers fetched")

        if st.button("Fetch OpenAlex", use_container_width=True):
            with st.spinner("Fetching from OpenAlex…"):
                oa_papers = fetch_openalex(strings.get("openalex",""))
            st.session_state["_oa"] = oa_papers
            st.success(f"{len(oa_papers)} papers from OpenAlex")
        if "_oa" in st.session_state:
            st.caption(f"OpenAlex: {len(st.session_state['_oa'])} papers fetched")

    # ── Manual uploads ───────────────────────────────────────
    with col2:
        st.markdown("**✋ Manual uploads**")
        sc_file  = st.file_uploader("Scopus export (CSV)", type=["csv"],  key="sc_upload")
        wos_file = st.file_uploader("WOS export (RIS/TXT)", type=["ris","txt"], key="wos_upload")

    # ── Merge & deduplicate ──────────────────────────────────
    st.divider()
    if st.button("🔗 Merge all sources & deduplicate", type="primary", use_container_width=True):
        all_papers = []
        all_papers += st.session_state.get("_pm", [])
        all_papers += st.session_state.get("_oa", [])
        if sc_file:
            content = sc_file.read().decode("utf-8", errors="replace")
            all_papers += parse_scopus_csv(content)
        if wos_file:
            content = wos_file.read().decode("utf-8", errors="replace")
            all_papers += parse_ris(content, "WOS") if wos_file.name.endswith(".ris") else parse_wos(content)

        if not all_papers:
            st.warning("No papers found. Fetch from PubMed/OpenAlex or upload files first.")
        else:
            st.session_state.papers_raw     = all_papers
            st.session_state.papers_deduped = deduplicate(all_papers)
            st.rerun()

# ════════════════════════════════════════════════════════════
# PHASE 2 — AI screening
# ════════════════════════════════════════════════════════════
elif not st.session_state.screening_done:
    papers = st.session_state.papers_deduped
    raw_n  = len(st.session_state.papers_raw or [])
    dedup_n= len(papers)

    st.markdown("### 2 · AI screening")

    # Source breakdown
    from collections import Counter
    src_counts = Counter(p.get("source","?") for p in papers)
    cols = st.columns(len(src_counts) + 2)
    cols[0].metric("Total retrieved", raw_n)
    cols[1].metric("After dedup", dedup_n)
    for i, (src, cnt) in enumerate(src_counts.items()):
        cols[i+2].metric(src, cnt)

    st.info(
        f"Gemini will screen all **{dedup_n} papers** in batches of 5. "
        f"Estimated time: ~{max(1, dedup_n//5 * 7 // 60)} min "
        f"({dedup_n//5 + 1} API calls at 10 req/min)."
    )

    if st.button("▶ Run AI Screening", type="primary", use_container_width=True):
        model   = genai.GenerativeModel(model_name=MODEL_NAME)
        results = {}
        batch_size = 5
        batches    = [papers[i:i+batch_size] for i in range(0, len(papers), batch_size)]
        progress   = st.progress(0, text="Starting AI screening…")
        status_box = st.empty()

        for b_idx, batch in enumerate(batches):
            status_box.caption(f"Screening papers {b_idx*batch_size+1}–{min((b_idx+1)*batch_size, len(papers))} of {len(papers)}…")
            decisions = screen_batch(batch, pico, model)
            for item in decisions:
                global_idx = b_idx * batch_size + item["id"]
                results[global_idx] = {
                    "ai_decision": item.get("decision","UNCERTAIN"),
                    "ai_reason":   item.get("reason",""),
                    "human_decision": None,
                }
            progress.progress((b_idx+1)/len(batches), text=f"Screened {min((b_idx+1)*batch_size, len(papers))}/{len(papers)} papers")
            if b_idx < len(batches) - 1:
                time.sleep(7)  # stay under 10 RPM

        st.session_state.screening_results = results
        st.session_state.screening_done    = True
        st.session_state.review_index      = 0
        progress.progress(1.0, text="✅ AI screening complete!")
        time.sleep(0.5)
        st.rerun()

# ════════════════════════════════════════════════════════════
# PHASE 3 — Human review
# ════════════════════════════════════════════════════════════
else:
    papers  = st.session_state.papers_deduped
    results = st.session_state.screening_results
    n_total = len(papers)

    # Summary counts
    ai_inc  = sum(1 for r in results.values() if r["ai_decision"] == "INCLUDE")
    ai_exc  = sum(1 for r in results.values() if r["ai_decision"] == "EXCLUDE")
    ai_unc  = sum(1 for r in results.values() if r["ai_decision"] == "UNCERTAIN")
    h_done  = sum(1 for r in results.values() if r["human_decision"] is not None)
    h_inc   = sum(1 for r in results.values() if r["human_decision"] == "INCLUDE")
    h_exc   = sum(1 for r in results.values() if r["human_decision"] == "EXCLUDE")

    st.markdown("### 3 · Human review")

    # Progress bar
    st.progress(h_done / n_total if n_total else 0,
                text=f"Reviewed {h_done} of {n_total} papers")

    # Summary metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total",     n_total)
    c2.metric("AI: Include",  ai_inc)
    c3.metric("AI: Uncertain",ai_unc)
    c4.metric("AI: Exclude",  ai_exc)
    c5.metric("Human done",   h_done)

    st.divider()

    # Filter tabs
    tab_all, tab_inc, tab_unc, tab_exc = st.tabs([
        f"All ({n_total})",
        f"✅ Include ({ai_inc})",
        f"⚠️ Uncertain ({ai_unc})",
        f"❌ Exclude ({ai_exc})",
    ])

    def render_paper_card(idx, paper, result, tab_key):
        decision = result["ai_decision"]
        badge_cls = {"INCLUDE":"ai-include","EXCLUDE":"ai-exclude","UNCERTAIN":"ai-uncertain"}[decision]
        badge_txt = {"INCLUDE":"✅ AI: Include","EXCLUDE":"❌ AI: Exclude","UNCERTAIN":"⚠️ AI: Uncertain"}[decision]

        src_badges = f'<span class="src-badge">{paper.get("source","?")}</span>'
        meta = f'{paper.get("authors","—")} · {paper.get("year","?")} · {paper.get("journal","?")}'

        st.markdown(f"""
        <div class="paper-card">
            <div class="paper-title">{paper.get('title','No title')}</div>
            <div class="paper-meta">{src_badges} {meta}</div>
            <div class="paper-abstract">{paper.get('abstract','No abstract available.')}</div>
            <span class="{badge_cls}">{badge_txt}</span>
            <div class="ai-reason">Reason: {result.get('ai_reason','')}</div>
        </div>
        """, unsafe_allow_html=True)

        h = result.get("human_decision")
        lbl = f"Human: **{'✅ Included' if h=='INCLUDE' else ('❌ Excluded' if h=='EXCLUDE' else 'Not reviewed')}**" if h else "Your decision:"
        st.caption(lbl)

        b1, b2, b3 = st.columns(3)
        if b1.button("✅ Include", key=f"inc_{tab_key}_{idx}", use_container_width=True):
            st.session_state.screening_results[idx]["human_decision"] = "INCLUDE"
            st.rerun()
        if b2.button("❌ Exclude", key=f"exc_{tab_key}_{idx}", use_container_width=True):
            st.session_state.screening_results[idx]["human_decision"] = "EXCLUDE"
            st.rerun()
        if b3.button("↩️ Undo",   key=f"undo_{tab_key}_{idx}", use_container_width=True, disabled=(h is None)):
            st.session_state.screening_results[idx]["human_decision"] = None
            st.rerun()
        st.divider()

    with tab_all:
        for idx, paper in enumerate(papers):
            render_paper_card(idx, paper, results.get(idx,{}), "all")

    with tab_inc:
        for idx, paper in enumerate(papers):
            r = results.get(idx,{})
            if r.get("ai_decision") == "INCLUDE":
                render_paper_card(idx, paper, r, "inc")

    with tab_unc:
        for idx, paper in enumerate(papers):
            r = results.get(idx,{})
            if r.get("ai_decision") == "UNCERTAIN":
                render_paper_card(idx, paper, r, "unc")

    with tab_exc:
        for idx, paper in enumerate(papers):
            r = results.get(idx,{})
            if r.get("ai_decision") == "EXCLUDE":
                render_paper_card(idx, paper, r, "exc")

    # ── Final summary ────────────────────────────────────────
    if h_done == n_total:
        st.success(f"✅ All papers reviewed! **{h_inc} included**, {h_exc} excluded.")
        st.info("✅ Step 3 complete — proceed to Step 4: Quality Appraisal.")
        col_exp, col_next = st.columns([2,1])
        with col_exp:
            # Export included papers as CSV
            inc_papers = [
                {**papers[i], "ai_decision": results[i]["ai_decision"], "human_decision": results[i]["human_decision"]}
                for i in range(n_total) if results.get(i,{}).get("human_decision") == "INCLUDE"
            ]
            csv_out = io.StringIO()
            if inc_papers:
                writer = csv.DictWriter(csv_out, fieldnames=inc_papers[0].keys())
                writer.writeheader()
                writer.writerows(inc_papers)
                st.download_button(
                    "⬇️ Download included papers (CSV)",
                    csv_out.getvalue(),
                    file_name="sr_included_papers.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
