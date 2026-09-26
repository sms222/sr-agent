import streamlit as st
import json
import requests
import google.generativeai as genai

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="SR Agent · PICO Refiner",
    page_icon="🔬",
    layout="centered",
)

MODEL_NAME = "gemini-2.5-flash"

# ── Styles ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Lora:ital,wght@0,600;1,400&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Lora', serif; }
.main-title { font-family:'Lora',serif; font-size:1.7rem; font-weight:600; color:#0A3D2E; margin-bottom:0.15rem; }
.main-subtitle { font-size:0.85rem; color:#64748B; margin-bottom:1.5rem; }
.pico-card { background:#FFFFFF; border:1.5px solid #BBD8C8; border-radius:10px; padding:1.4rem 1.75rem 1.2rem; margin:1.25rem 0; }
.pico-card-title { font-family:'Lora',serif; font-size:0.95rem; font-weight:600; color:#0A3D2E; margin-bottom:1rem; }
.pico-row { display:flex; gap:1rem; margin-bottom:0.55rem; align-items:baseline; }
.pico-label { font-size:0.7rem; font-weight:600; color:#0D5C4E; width:140px; flex-shrink:0; text-transform:uppercase; letter-spacing:0.05em; }
.pico-value { font-size:0.88rem; color:#1E293B; line-height:1.55; }
.formal-q { background:#F0FDF7; border-left:3px solid #0D5C4E; padding:0.7rem 1rem; margin-top:1rem; border-radius:0 6px 6px 0; font-size:0.88rem; font-style:italic; color:#1E293B; line-height:1.65; }
.scope-label { font-size:0.7rem; font-weight:600; color:#64748B; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.25rem; }
.step-badge { display:inline-block; background:#E8F5EE; color:#0D5C4E; font-size:0.7rem; font-weight:600; padding:0.2rem 0.6rem; border-radius:20px; letter-spacing:0.03em; margin-bottom:0.5rem; }
.mcq-label { font-size:0.75rem; color:#64748B; margin-bottom:0.4rem; margin-top:0.75rem; }
</style>
""", unsafe_allow_html=True)

# ── System prompt ────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a systematic review methodologist. Your job is to extract a clear PICO from the user's research idea.

Be direct and brief. No filler phrases — no "great", "certainly", "that's a good question", "of course", "excellent", "happy to help", "understood". Do not echo back what the user just said. Just ask the next question or confirm.

Steps:
1. Ask ONE question at a time to fill in missing PICO elements:
   P – Population / Problem
   I – Intervention (or Exposure for observational questions)
   C – Comparator (may be "none")
   O – Outcome(s)
   Optional: study design preference
2. Adapt framework as needed:
   - Qualitative → SPIDER
   - Animal/preclinical → note SYRCLE will be used later
   - Scoping review → PICOS with broad framing
3. Once all elements are clear, show a PICO summary and ask the user to confirm.
4. On confirmation, output ONLY this JSON as the last thing in your reply:
   {"status": "confirmed", "P": "...", "I": "...", "C": "...", "O": "...", "study_design": "...", "formal_question": "..."}

MCQ rule: When your question has 2–4 clear predefined options, append this block at the end:
<<<MCQ>>>{"question": "label", "options": ["A", "B", "C", "Other (I'll type)"]}<<<END>>>
Always include "Other (I'll type)" as last option.
Do not use MCQ for open-ended questions (drug names, specific outcomes, etc.).
Never combine MCQ block and confirmed PICO JSON in the same message.

The formal_question must be a complete, publication-ready research question.
Output the confirmed JSON only once the user has explicitly confirmed.
""".strip()

MCQ_START = "<<<MCQ>>>"
MCQ_END   = "<<<END>>>"

# ── Scope helpers ────────────────────────────────────────────
def check_pubmed_scope(pico):
    skip_keys = {"status", "study_design", "formal_question"}
    no_comp   = {"none","no comparator","no intervention","placebo","n/a","na","not applicable","no control"}
    terms = [
        v.strip() for k, v in pico.items()
        if v and k not in skip_keys
        and not (k == "C" and v.lower().strip() in no_comp)
    ]
    query = " AND ".join(terms)
    try:
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db":"pubmed","term":query,"retmode":"json","retmax":0},
            timeout=10,
        )
        return int(r.json()["esearchresult"]["count"]), query
    except Exception:
        return None, query

def check_openalex_scope(pico):
    skip_keys = {"status", "study_design", "formal_question"}
    no_comp   = {"none","no comparator","no intervention","placebo","n/a","na","not applicable","no control"}
    terms = [
        v.strip() for k, v in pico.items()
        if v and k not in skip_keys
        and not (k == "C" and v.lower().strip() in no_comp)
    ]
    query = " ".join(terms)
    try:
        r = requests.get(
            "https://api.openalex.org/works",
            params={"search":query,"per-page":1},
            headers={"User-Agent":"SR-Agent/1.0 (mailto:sr-agent@ukm.edu.my)"},
            timeout=10,
        )
        return r.json().get("meta",{}).get("count")
    except Exception:
        return None

def scope_verdict(pubmed, openalex):
    counts = [c for c in [pubmed, openalex] if c is not None]
    if not counts:
        return "⚠️","warning","Scope check unavailable — check your connection."
    avg = sum(counts)/len(counts)
    if avg == 0:     return "🔴","error",   "No results found. Question may be too narrow or terminology needs adjusting."
    elif avg < 20:   return "🟡","warning", f"~{int(avg)} results — consider broadening population or outcome terms."
    elif avg <= 500: return "🟢","success", f"~{int(avg)} results — good scope for a systematic review."
    elif avg <= 2000:return "🟡","warning", f"~{int(avg)} results — consider tightening comparator or outcome."
    else:            return "🔴","error",   f"~{int(avg)} results — too broad. Narrow the question before proceeding."

def parse_mcq(reply: str):
    if MCQ_START in reply and MCQ_END in reply:
        start = reply.index(MCQ_START)
        text  = reply[:start].strip()
        raw   = reply[start+len(MCQ_START): reply.index(MCQ_END)].strip()
        try:
            return text, json.loads(raw).get("options",[])
        except Exception:
            return reply, []
    return reply, []

# ── send_message — uses explicit history, no ChatSession ─────
def send_message(user_text: str) -> bool:
    st.session_state.messages.append({"role":"user","content":user_text})
    st.session_state.pending_mcq = []

    # Build full history for Gemini (plain dicts — survive reruns reliably)
    model   = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=SYSTEM_PROMPT)
    history = st.session_state.gemini_history + [{"role":"user","parts":[user_text]}]

    try:
        response = model.generate_content(history)
        reply    = response.text
    except Exception as e:
        st.session_state.messages.pop()
        err = str(e).lower()
        if "resourceexhausted" in err or "quota" in err or "429" in err:
            st.session_state["_err"] = "⏳ Rate limit reached (10 req/min). Wait 30 seconds and try again."
        else:
            st.session_state["_err"] = f"Gemini error: {e}"
        return False

    # Persist history as plain dicts
    st.session_state.gemini_history.append({"role":"user","parts":[user_text]})
    st.session_state.gemini_history.append({"role":"model","parts":[reply]})

    # Confirmed PICO?
    marker = '{"status": "confirmed"'
    if marker in reply:
        json_start = reply.index(marker)
        preamble   = reply[:json_start].strip()
        st.session_state.messages.append({"role":"assistant","content": preamble or "PICO confirmed."})
        try:
            pico_result = json.loads(reply[json_start:])
        except json.JSONDecodeError:
            chunk = reply[json_start:]
            pico_result = json.loads(chunk[:chunk.rfind("}")+1])
        st.session_state.pico = pico_result
        pm_count, pm_query = check_pubmed_scope(pico_result)
        oa_count           = check_openalex_scope(pico_result)
        st.session_state.pm_count = pm_count
        st.session_state.oa_count = oa_count
        st.session_state.pm_query = pm_query
        return True

    # MCQ or plain text
    text, options = parse_mcq(reply)
    if options:
        st.session_state.messages.append({"role":"assistant","content":text})
        st.session_state.pending_mcq = options
    else:
        st.session_state.messages.append({"role":"assistant","content":reply})
    return True

# ── API key ──────────────────────────────────────────────────
def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY","")
        if key:
            return key, True
    except Exception:
        pass
    return "", False

secret_key, from_secrets = get_api_key()

# ── Session state ────────────────────────────────────────────
defaults = {
    "messages":       [],
    "gemini_history": [],   # explicit history — replaces ChatSession
    "pico":           None,
    "pm_count":       None,
    "oa_count":       None,
    "pm_query":       None,
    "pending_mcq":    [],
    "_err":           None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 SR Agent")
    st.caption("AI-assisted systematic review pipeline")
    st.divider()
    if from_secrets:
        api_key = secret_key
    else:
        api_key = st.text_input("Gemini API Key", type="password", placeholder="AIza…",
                                help="Get a free key at aistudio.google.com/apikey")
        st.caption("Add to Streamlit Secrets as `GEMINI_API_KEY` to avoid re-entering.")
    st.caption(f"Model: `{MODEL_NAME}` · Free tier: 500 req/day")
    st.divider()
    for num, name, active in [("1","PICO Refiner",True),("2","Search String Builder",False),("3","Abstract Screener",False),("4","Quality Appraisal",False),("5","Data Extraction",False)]:
        if active:
            st.markdown(f"**→ Step {num}: {name}**")
        else:
            st.markdown(f"<span style='color:#94A3B8'>Step {num}: {name}</span>", unsafe_allow_html=True)
    st.divider()
    if st.button("↺ Start over", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

# ── Main ─────────────────────────────────────────────────────
st.markdown('<div class="step-badge">Step 1 of 5</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Research Question Refiner</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Describe your research idea and the agent will refine it into a structured PICO.</div>', unsafe_allow_html=True)

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to begin. Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).")
    st.stop()

genai.configure(api_key=api_key)

# Welcome message (once only)
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": "What is your research idea or question?"
    })

# Show persistent error if any
if st.session_state.get("_err"):
    st.warning(st.session_state.pop("_err"))

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# PICO confirmed
if st.session_state.pico:
    pico = st.session_state.pico
    rows = [("P — Population",pico.get("P","—")),("I — Intervention",pico.get("I","—")),("C — Comparator",pico.get("C","—")),("O — Outcome",pico.get("O","—"))]
    if pico.get("study_design"):
        rows.append(("Study Design", pico["study_design"]))
    rows_html = "".join(f'<div class="pico-row"><span class="pico-label">{l}</span><span class="pico-value">{v}</span></div>' for l,v in rows)
    st.markdown(f'<div class="pico-card"><div class="pico-card-title">✅ Confirmed PICO</div>{rows_html}<div class="formal-q">{pico.get("formal_question","")}</div></div>', unsafe_allow_html=True)

    pm, oa = st.session_state.pm_count, st.session_state.oa_count
    if pm is not None or oa is not None:
        st.markdown("**Scope check**")
        c1, c2, c3 = st.columns(3)
        c1.metric("PubMed hits",   f"{pm:,}" if pm is not None else "—")
        c2.metric("OpenAlex hits", f"{oa:,}" if oa is not None else "—")
        emoji, kind, vtext = scope_verdict(pm, oa)
        with c3:
            st.markdown(f"<div class='scope-label'>Verdict</div>{emoji}", unsafe_allow_html=True)
        getattr(st, kind)(vtext)

    st.divider()
    ca, cb = st.columns([2,1])
    ca.info("✅ Step 1 complete — PICO locked in.")
    if cb.button("Continue to Step 2 →", use_container_width=True, type="primary"):
        st.switch_page("pages/2_Search_String_Builder.py")
    st.stop()

# MCQ buttons
if st.session_state.pending_mcq:
    st.markdown('<div class="mcq-label">Choose an option or type your own below</div>', unsafe_allow_html=True)
    for i, option in enumerate(st.session_state.pending_mcq):
        if st.button(option, key=f"mcq_{i}", use_container_width=True):
            send_message(option)
            st.rerun()

# Free-text input
if prompt := st.chat_input("Type your research idea or answer…"):
    send_message(prompt)
    st.rerun()
