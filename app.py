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
.pico-card {
    background: #FFFFFF;
    border: 1.5px solid #BBD8C8;
    border-radius: 10px;
    padding: 1.4rem 1.75rem 1.2rem;
    margin: 1.25rem 0;
}
.pico-card-title {
    font-family: 'Lora', serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #0A3D2E;
    margin-bottom: 1rem;
}
.pico-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 0.55rem;
    align-items: baseline;
}
.pico-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #0D5C4E;
    width: 140px;
    flex-shrink: 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.pico-value {
    font-size: 0.88rem;
    color: #1E293B;
    line-height: 1.55;
}
.formal-q {
    background: #F0FDF7;
    border-left: 3px solid #0D5C4E;
    padding: 0.7rem 1rem;
    margin-top: 1rem;
    border-radius: 0 6px 6px 0;
    font-size: 0.88rem;
    font-style: italic;
    color: #1E293B;
    line-height: 1.65;
}
.scope-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
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
.mcq-label {
    font-size: 0.75rem;
    color: #64748B;
    margin-bottom: 0.4rem;
    margin-top: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# ── System prompt ────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert systematic review methodologist helping a researcher
refine a rough research idea into a well-structured PICO or PECO framework.

Your task:
1. Listen to the user's rough question or idea.
2. Ask ONE targeted clarifying question at a time to fill in missing elements:
     P – Population / Problem
     I – Intervention (or Exposure for observational/environmental questions)
     C – Comparator / Control (if applicable; may be "none" for some designs)
     O – Outcome(s) of interest
     Optional: preferred study designs (RCT, cohort, qualitative, animal, etc.)
3. Adapt the framework as needed:
   - Qualitative research → use SPIDER (Sample, PI, Design, Evaluation, Research type)
   - Animal / preclinical → note that SYRCLE will be used for appraisal later
   - Scoping reviews → PICOS with a broad scope framing
4. Once all elements are clear, present a structured PICO summary for the user to confirm.
5. When the user confirms (or approves with minor edits), output ONLY the confirmed PICO JSON
   as the very last thing in your reply — nothing after it:
   {"status": "confirmed", "P": "...", "I": "...", "C": "...", "O": "...", "study_design": "...", "formal_question": "..."}

IMPORTANT — MCQ questions:
When your clarifying question has 2–4 clear predefined answers (e.g. study design type,
age group, yes/no, broad category choice), append this block at the very end of your message:
<<<MCQ>>>{"question": "Short question label", "options": ["Option A", "Option B", "Option C", "Other (I'll type)"]}<<<END>>>

Always include "Other (I'll type)" as the last MCQ option.
For open-ended questions where free text is needed (e.g. specific drug names, outcome details),
ask as plain text with NO MCQ block.
Never output both an MCQ block and the confirmed PICO JSON in the same message.

Rules:
- Ask ONE question at a time — never a list.
- Be conversational, concise, and encouraging.
- If the question seems too broad or narrow, flag it gently.
- The formal_question must read as a complete, publication-ready research question.
- Only output the confirmed PICO JSON once the user has explicitly confirmed.
""".strip()

MCQ_START = "<<<MCQ>>>"
MCQ_END   = "<<<END>>>"

# ── Scope helpers ────────────────────────────────────────────
def check_pubmed_scope(pico):
    skip = {"status", "study_design", "formal_question"}
    terms = [f'("{v}"[Title/Abstract])' for k, v in pico.items() if v and k not in skip]
    query = " AND ".join(terms)
    try:
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmode": "json", "retmax": 0},
            timeout=10,
        )
        return int(r.json()["esearchresult"]["count"]), query
    except Exception:
        return None, query

def check_openalex_scope(pico):
    skip = {"status", "study_design", "formal_question"}
    query = " ".join(v for k, v in pico.items() if v and k not in skip)
    try:
        r = requests.get(
            "https://api.openalex.org/works",
            params={"search": query, "per-page": 1},
            headers={"User-Agent": "SR-Agent/1.0 (mailto:sr-agent@ukm.edu.my)"},
            timeout=10,
        )
        return r.json().get("meta", {}).get("count")
    except Exception:
        return None

def scope_verdict(pubmed, openalex):
    counts = [c for c in [pubmed, openalex] if c is not None]
    if not counts:
        return "⚠️", "warning", "Scope check unavailable — check your connection."
    avg = sum(counts) / len(counts)
    if avg == 0:
        return "🔴", "error", "No results found. Question may be too narrow or terminology needs adjusting."
    elif avg < 20:
        return "🟡", "warning", f"~{int(avg)} results on average — consider broadening population or outcome terms."
    elif avg <= 500:
        return "🟢", "success", f"~{int(avg)} results on average — good scope for a systematic review."
    elif avg <= 2000:
        return "🟡", "warning", f"~{int(avg)} results on average — consider tightening comparator or outcome."
    else:
        return "🔴", "error", f"~{int(avg)} results on average — too broad. Narrow the question before proceeding."

def parse_mcq(reply: str):
    """Extract plain text + MCQ options from a reply, if present."""
    if MCQ_START in reply and MCQ_END in reply:
        start = reply.index(MCQ_START)
        end   = reply.index(MCQ_END) + len(MCQ_END)
        text  = reply[:start].strip()
        raw   = reply[start + len(MCQ_START): reply.index(MCQ_END)].strip()
        try:
            mcq = json.loads(raw)
            return text, mcq.get("options", [])
        except Exception:
            return reply, []
    return reply, []

def send_message(user_text: str):
    """Send a message to Gemini, parse the reply, update session state."""
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.pending_mcq = []   # clear pending options

    response = st.session_state.chat.send_message(user_text)
    reply    = response.text

    # Check confirmed PICO
    pico_marker = '{"status": "confirmed"'
    if pico_marker in reply:
        json_start = reply.index(pico_marker)
        preamble   = reply[:json_start].strip()
        display    = preamble or "Your PICO is confirmed — here's the summary."
        st.session_state.messages.append({"role": "assistant", "content": display})
        try:
            pico_result = json.loads(reply[json_start:])
        except json.JSONDecodeError:
            chunk = reply[json_start:]
            pico_result = json.loads(chunk[: chunk.rfind("}") + 1])
        st.session_state.pico = pico_result

        # Scope check
        pm_count, pm_query = check_pubmed_scope(pico_result)
        oa_count           = check_openalex_scope(pico_result)
        st.session_state.pm_count = pm_count
        st.session_state.oa_count = oa_count
        st.session_state.pm_query = pm_query
        return

    # Check MCQ
    text, options = parse_mcq(reply)
    if options:
        st.session_state.messages.append({"role": "assistant", "content": text})
        st.session_state.pending_mcq = options
    else:
        st.session_state.messages.append({"role": "assistant", "content": reply})

# ── API key resolution ───────────────────────────────────────
# Priority: Streamlit Cloud secrets → sidebar input
def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key, True   # (key, from_secrets)
    except Exception:
        pass
    return "", False

secret_key, from_secrets = get_api_key()

# ── Session state ────────────────────────────────────────────
defaults = {
    "messages":    [],
    "chat":        None,
    "pico":        None,
    "pm_count":    None,
    "oa_count":    None,
    "pm_query":    None,
    "pending_mcq": [],
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
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIza…",
            help="Get a free key at aistudio.google.com/apikey",
        )
        st.caption(
            "To avoid entering this every time, add it to your app's "
            "**Secrets** in the Streamlit Cloud dashboard:\n\n"
            "```\nGEMINI_API_KEY = \"AIza...\"\n```"
        )

    st.caption(f"Model: `{MODEL_NAME}` · Free tier: 500 req/day")
    st.divider()

    steps = [
        ("1", "PICO Refiner",         True),
        ("2", "Search String Builder", False),
        ("3", "Abstract Screener",     False),
        ("4", "Quality Appraisal",     False),
        ("5", "Data Extraction",       False),
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

    st.divider()
    if st.button("↺ Start over", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

# ── Main ─────────────────────────────────────────────────────
st.markdown('<div class="step-badge">Step 1 of 5</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Research Question Refiner</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Describe your research idea and the agent will '
    'refine it into a structured PICO.</div>',
    unsafe_allow_html=True,
)

if not api_key:
    st.info(
        "Enter your Gemini API key in the sidebar to begin. "
        "Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)."
    )
    st.stop()

# Configure Gemini and init chat once
genai.configure(api_key=api_key)

if st.session_state.chat is None:
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_PROMPT,
    )
    st.session_state.chat = model.start_chat(history=[])
    welcome = (
        "Hello! Tell me your research idea — as rough as you like. "
        "What topic or clinical problem are you hoping to explore in your systematic review?"
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome})

# ── Render chat history ──────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── PICO confirmed — show card + scope ───────────────────────
if st.session_state.pico:
    pico = st.session_state.pico
    rows = [
        ("P — Population",   pico.get("P", "—")),
        ("I — Intervention", pico.get("I", "—")),
        ("C — Comparator",   pico.get("C", "—")),
        ("O — Outcome",      pico.get("O", "—")),
    ]
    if pico.get("study_design"):
        rows.append(("Study Design", pico["study_design"]))

    rows_html = "".join(
        f'<div class="pico-row">'
        f'<span class="pico-label">{label}</span>'
        f'<span class="pico-value">{value}</span>'
        f'</div>'
        for label, value in rows
    )
    st.markdown(f"""
    <div class="pico-card">
        <div class="pico-card-title">✅ Confirmed PICO</div>
        {rows_html}
        <div class="formal-q">{pico.get('formal_question', '')}</div>
    </div>
    """, unsafe_allow_html=True)

    pm, oa = st.session_state.pm_count, st.session_state.oa_count
    if pm is not None or oa is not None:
        st.markdown("**Scope check**")
        col1, col2, col3 = st.columns(3)
        col1.metric("PubMed hits",   f"{pm:,}" if pm is not None else "—")
        col2.metric("OpenAlex hits", f"{oa:,}" if oa is not None else "—")
        emoji, kind, verdict_text = scope_verdict(pm, oa)
        with col3:
            st.markdown(
                f"<div class='scope-label'>Verdict</div>{emoji}",
                unsafe_allow_html=True,
            )
        if kind == "success":
            st.success(verdict_text)
        elif kind == "warning":
            st.warning(verdict_text)
        else:
            st.error(verdict_text)

    st.info("✅ Step 1 complete — PICO locked in. Step 2 (Search String Builder) coming next.")
    st.stop()

# ── MCQ buttons (shown when Gemini offered options) ──────────
if st.session_state.pending_mcq:
    st.markdown('<div class="mcq-label">Choose an option or type your own below</div>',
                unsafe_allow_html=True)
    cols = st.columns(len(st.session_state.pending_mcq))
    for i, option in enumerate(st.session_state.pending_mcq):
        if cols[i].button(option, key=f"mcq_{i}", use_container_width=True):
            with st.spinner(""):
                send_message(option)
            st.rerun()

# ── Free-text chat input ─────────────────────────────────────
if prompt := st.chat_input("Describe your research idea or type your answer…"):
    with st.spinner(""):
        send_message(prompt)
    st.rerun()
