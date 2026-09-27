import streamlit as st
import json
import requests
import google.generativeai as genai
from openai import OpenAI

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="SR Agent · PICO Refiner",
    page_icon="🔬",
    layout="centered",
)

# ── Provider / model registry ────────────────────────────────
PROVIDERS = {
    "groq_llama33": {
        "label":      "GPT OSS 120B · Groq",
        "provider":   "Groq",
        "model_id":   "openai/gpt-oss-120b",
        "limits":     "30 RPM · 1,000/day",
        "secret_key": "GROQ_API_KEY",
        "base_url":   "https://api.groq.com/openai/v1",
        "type":       "openai_compat",
        "badge":      "#6366F1",
    },
    "groq_llama4": {
        "label":      "Qwen 3.6 27B · Groq",
        "provider":   "Groq",
        "model_id":   "qwen/qwen3.6-27b",
        "limits":     "30 RPM · 1,000/day",
        "secret_key": "GROQ_API_KEY",
        "base_url":   "https://api.groq.com/openai/v1",
        "type":       "openai_compat",
        "badge":      "#8B5CF6",
    },
    "cerebras_llama33": {
        "label":      "Llama 3.3 70B · Cerebras",
        "provider":   "Cerebras",
        "model_id":   "llama-3.3-70b",
        "limits":     "30 RPM · 14,400/day",
        "secret_key": "CEREBRAS_API_KEY",
        "base_url":   "https://api.cerebras.ai/v1",
        "type":       "openai_compat",
        "badge":      "#059669",
    },
    "gemini_flash": {
        "label":      "Gemini 2.5 Flash · Google",
        "provider":   "Google",
        "model_id":   "gemini-2.5-flash",
        "limits":     "10 RPM · 500/day",
        "secret_key": "GEMINI_API_KEY",
        "base_url":   None,
        "type":       "gemini",
        "badge":      "#0D5C4E",
    },
}

DEFAULT_PROVIDER = "groq_llama33"

# ── Styles ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Lora:ital,wght@0,600;1,400&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
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
.model-badge { display:inline-flex; align-items:center; gap:0.35rem; font-size:0.7rem; font-weight:600; padding:0.2rem 0.65rem; border-radius:20px; color:#fff; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

# ── System prompt ────────────────────────────────────────────
SYSTEM_PROMPT = """You are a systematic review methodologist. Extract a clear PICO from the user's research idea.

Be direct and brief. No filler — no "great", "certainly", "of course", "excellent", "understood". Do not echo back what the user said. Ask the next question or confirm, nothing else.

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
Always include "Other (I'll type)" as last option. No MCQ for open-ended questions.
Never combine MCQ block and confirmed PICO JSON in the same message.
The formal_question must be a complete, publication-ready research question.
Output the confirmed JSON only once the user has explicitly confirmed."""

MCQ_START = "<<<MCQ>>>"
MCQ_END   = "<<<END>>>"

# ── LLM abstraction ──────────────────────────────────────────
def call_llm(history: list, provider_key: str, api_key: str) -> str:
    """
    history: list of {"role": "user"|"assistant", "content": "..."}
    Returns the model's reply as a string.
    """
    cfg = PROVIDERS[provider_key]

    if cfg["type"] == "openai_compat":
        client   = OpenAI(base_url=cfg["base_url"], api_key=api_key)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        response = client.chat.completions.create(
            model=cfg["model_id"],
            messages=messages,
            max_tokens=1000,
        )
        return response.choices[0].message.content

    elif cfg["type"] == "gemini":
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=cfg["model_id"],
            system_instruction=SYSTEM_PROMPT,
        )
        # Convert to Gemini history format
        gemini_history = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
            for m in history
        ]
        return model.generate_content(gemini_history).text

    raise ValueError(f"Unknown provider type: {cfg['type']}")

# ── Scope helpers ────────────────────────────────────────────
def check_pubmed_scope(pico):
    skip  = {"status","study_design","formal_question"}
    nocomp= {"none","no comparator","no intervention","placebo","n/a","na","not applicable","no control"}
    terms = [v.strip() for k,v in pico.items() if v and k not in skip and not (k=="C" and v.lower().strip() in nocomp)]
    query = " AND ".join(terms)
    try:
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                         params={"db":"pubmed","term":query,"retmode":"json","retmax":0},timeout=10)
        return int(r.json()["esearchresult"]["count"]), query
    except Exception:
        return None, query

def check_openalex_scope(pico):
    skip  = {"status","study_design","formal_question"}
    nocomp= {"none","no comparator","no intervention","placebo","n/a","na","not applicable","no control"}
    terms = [v.strip() for k,v in pico.items() if v and k not in skip and not (k=="C" and v.lower().strip() in nocomp)]
    try:
        r = requests.get("https://api.openalex.org/works",
                         params={"search":" ".join(terms),"per-page":1},
                         headers={"User-Agent":"SR-Agent/1.0"},timeout=10)
        return r.json().get("meta",{}).get("count")
    except Exception:
        return None

def scope_verdict(pm, oa):
    counts = [c for c in [pm,oa] if c is not None]
    if not counts: return "⚠️","warning","Scope check unavailable."
    avg = sum(counts)/len(counts)
    if avg==0:      return "🔴","error",  "No results — question may be too narrow."
    elif avg<20:    return "🟡","warning",f"~{int(avg)} results — consider broadening."
    elif avg<=500:  return "🟢","success",f"~{int(avg)} results — good scope."
    elif avg<=2000: return "🟡","warning",f"~{int(avg)} results — consider tightening."
    else:           return "🔴","error",  f"~{int(avg)} results — too broad."

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

# ── send_message ─────────────────────────────────────────────
def send_message(user_text: str) -> bool:
    st.session_state.messages.append({"role":"user","content":user_text})
    st.session_state.pending_mcq = []
    st.session_state.conversation_history.append({"role":"user","content":user_text})

    try:
        reply = call_llm(
            st.session_state.conversation_history,
            st.session_state.selected_provider,
            st.session_state.active_api_key,
        )
    except Exception as e:
        st.session_state.messages.pop()
        st.session_state.conversation_history.pop()
        err = str(e).lower()
        if "resourceexhausted" in err or "quota" in err or "429" in err or "rate" in err:
            st.session_state["_err"] = "⏳ Rate limit hit. Wait 30 seconds and retry — or switch to a different model in the sidebar."
        else:
            st.session_state["_err"] = f"Model error: {e}"
        return False

    st.session_state.conversation_history.append({"role":"assistant","content":reply})

    # Confirmed PICO? Match regardless of spacing in JSON
    import re as _re
    _match = _re.search(r'\{[^{]*"status"\s*:\s*"confirmed"', reply)
    if _match:
        json_start = _match.start()
        try:
            pico_result = json.loads(reply[json_start:])
        except json.JSONDecodeError:
            chunk = reply[json_start:]
            pico_result = json.loads(chunk[:chunk.rfind("}")+1])
        fq = pico_result.get("formal_question", "")
        display = (
            "Based on your responses, the suggested research question is:\n\n"
            f"*{fq}*\n\n"
            "Review the PICO summary below. When ready, continue to Step 2."
        )
        st.session_state.messages.append({"role":"assistant","content": display})
        st.session_state.pico = pico_result
        pm, pmq = check_pubmed_scope(pico_result)
        oa      = check_openalex_scope(pico_result)
        st.session_state.pm_count = pm
        st.session_state.oa_count = oa
        st.session_state.pm_query = pmq
        return True

    text, options = parse_mcq(reply)
    if options:
        st.session_state.messages.append({"role":"assistant","content":text})
        st.session_state.pending_mcq = options
    else:
        st.session_state.messages.append({"role":"assistant","content":reply})
    return True

# ── API key resolution (secrets → sidebar) ───────────────────
def resolve_key(secret_name: str, sidebar_key: str) -> str:
    try:
        k = st.secrets.get(secret_name,"")
        if k:
            return k
    except Exception:
        pass
    return st.session_state.get(sidebar_key,"")

# ── Session state ────────────────────────────────────────────
defaults = {
    "messages":             [],
    "conversation_history": [],
    "pico":                 None,
    "pm_count":             None,
    "oa_count":             None,
    "pm_query":             None,
    "pending_mcq":          [],
    "_err":                 None,
    "selected_provider":    DEFAULT_PROVIDER,
    "active_api_key":       "",
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 SR Agent")
    st.caption("AI-assisted systematic review pipeline")
    st.divider()

    # Model selector
    provider_labels = {k: v["label"] for k,v in PROVIDERS.items()}
    selected = st.selectbox(
        "Model",
        options=list(PROVIDERS.keys()),
        format_func=lambda k: PROVIDERS[k]["label"],
        index=list(PROVIDERS.keys()).index(st.session_state.selected_provider),
    )
    if selected != st.session_state.selected_provider:
        st.session_state.selected_provider    = selected
        st.session_state.conversation_history = []
        st.session_state.messages             = []
        st.session_state.pending_mcq          = []
        st.rerun()

    cfg = PROVIDERS[selected]
    st.caption(f"Limits: {cfg['limits']}")

    # API key for selected provider
    secret_name = cfg["secret_key"]
    sidebar_key = f"apikey_{selected}"
    api_key = resolve_key(secret_name, sidebar_key)
    if not api_key:
        api_key = st.text_input(
            f"{cfg['provider']} API Key",
            type="password",
            placeholder="Paste key here…",
            key=sidebar_key,
            help=f"Stored as `{secret_name}` in Streamlit Secrets to avoid re-entering.",
        )
    else:
        st.caption(f"🔑 {cfg['provider']} key loaded from Secrets")

    st.session_state.active_api_key = api_key

    st.divider()
    for num, name, active in [("1","PICO Refiner",True),("2","Search String Builder",False),("3","Abstract Screener",False),("4","Quality Appraisal",False),("5","Data Extraction",False)]:
        if active:
            st.markdown(f"**→ Step {num}: {name}**")
        else:
            st.markdown(f"<span style='color:#94A3B8'>Step {num}: {name}</span>", unsafe_allow_html=True)
    st.divider()
    if st.button("↺ Start over", use_container_width=True):
        for k,v in defaults.items():
            st.session_state[k] = v
        st.rerun()

# ── Main ─────────────────────────────────────────────────────
st.markdown('<div class="step-badge">Step 1 of 5</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Research Question Refiner</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Describe your research idea and the agent will refine it into a structured PICO.</div>', unsafe_allow_html=True)

# Model badge
badge_color = cfg["badge"]
st.markdown(
    f'<div class="model-badge" style="background:{badge_color}">'
    f'⚡ {cfg["label"]} &nbsp;·&nbsp; {cfg["limits"]}</div>',
    unsafe_allow_html=True,
)

if not api_key:
    provider_urls = {
        "Groq":      "https://console.groq.com/keys",
        "Cerebras":  "https://cloud.cerebras.ai",
        "Google":    "https://aistudio.google.com/apikey",
    }
    url = provider_urls.get(cfg["provider"],"#")
    st.info(f"Enter your {cfg['provider']} API key in the sidebar. Get a free key at [{url}]({url}).")
    st.stop()

# Welcome message
if not st.session_state.messages:
    st.session_state.messages.append({"role":"assistant","content":"What is your research idea or question?"})

# Persistent error
if st.session_state.get("_err"):
    st.warning(st.session_state.pop("_err"))

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# PICO confirmed card
if st.session_state.pico:
    pico = st.session_state.pico
    rows = [("P — Population",pico.get("P","—")),("I — Intervention",pico.get("I","—")),
            ("C — Comparator",pico.get("C","—")),("O — Outcome",pico.get("O","—"))]
    if pico.get("study_design"):
        rows.append(("Study Design",pico["study_design"]))
    rows_html = "".join(f'<div class="pico-row"><span class="pico-label">{l}</span><span class="pico-value">{v}</span></div>' for l,v in rows)
    st.markdown(f'<div class="pico-card"><div class="pico-card-title">✅ Confirmed PICO</div>{rows_html}<div class="formal-q">{pico.get("formal_question","")}</div></div>', unsafe_allow_html=True)

    pm, oa = st.session_state.pm_count, st.session_state.oa_count
    if pm is not None or oa is not None:
        st.markdown("**Scope check**")
        c1,c2,c3 = st.columns(3)
        c1.metric("PubMed hits",   f"{pm:,}" if pm is not None else "—")
        c2.metric("OpenAlex hits", f"{oa:,}" if oa is not None else "—")
        emoji,kind,vtext = scope_verdict(pm,oa)
        with c3:
            st.markdown(f"<div class='scope-label'>Verdict</div>{emoji}", unsafe_allow_html=True)
        getattr(st, kind)(vtext)

    st.divider()
    st.markdown("**Does this look right?**")
    col_edit, col_cont = st.columns(2)
    if col_edit.button("✏️ Edit PICO", use_container_width=True):
        st.session_state.pico = None
        st.session_state.messages.append({"role": "assistant", "content": "What would you like to change?"})
        st.rerun()
    if col_cont.button("✅ Continue to Step 2 →", use_container_width=True, type="primary"):
        try:
            st.switch_page("pages/2_Search_String_Builder.py")
        except Exception:
            st.error("Page not found — make sure pages/2_Search_String_Builder.py is pushed to your GitHub repo.")
    st.stop()

# MCQ buttons
if st.session_state.pending_mcq:
    st.markdown('<div class="mcq-label">Choose an option or type your own below</div>', unsafe_allow_html=True)
    for i, option in enumerate(st.session_state.pending_mcq):
        if st.button(option, key=f"mcq_{i}", use_container_width=True):
            send_message(option)
            st.rerun()

# Chat input
if prompt := st.chat_input("Type your research idea or answer…"):
    send_message(prompt)
    st.rerun()
