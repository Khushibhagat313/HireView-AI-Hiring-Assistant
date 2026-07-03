import streamlit as st
from src.state import HiringState

st.set_page_config(layout="wide", page_title="HireView | AI Hiring Assistant", page_icon="🔍")

# ─── Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stButton>button { border-radius: 6px; font-weight: 500; transition: 0.2s; }
.candidate-card { border-left: 4px solid #3B4F7A; padding: 4px 0; }
.score-chip { 
    display: inline-block; padding: 2px 10px; border-radius: 12px; 
    font-size: 13px; font-weight: 600; margin-bottom: 4px;
}
.score-high { background: #d4edda; color: #155724; }
.score-med  { background: #fff3cd; color: #856404; }
.score-low  { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

CATEGORY_LIST = [
    "Data Science", "Python Developer", "Java Developer", "SQL Developer",
    "DevOps", "Testing", "Web Designing", "React Developer", "Business Analyst",
    "Database", "ETL Developer", "Hadoop", "Workday", "DotNet Developer",
    "Oracle", "SAP Developer", "Automation Testing", "Network Security Engineer",
    "PMO", "Blockchain", "Mechanical Engineer", "Civil Engineer",
    "Health and fitness", "Arts", "Advocate",
]

# ─── Session State Init ──────────────────────────────────────────────────────
if "hiring_state" not in st.session_state:
    st.session_state.hiring_state = HiringState(
        job_title="",
        job_description_raw="",
        job_description_structured=None,
        candidates=[],
        selected_candidate=None,
        conversation_history=[],
        current_suggestions=[],
        filters_applied={"threshold": 0.35},
        feedback_cache={},
        total_results=0
    )
if "search_submitted" not in st.session_state:
    st.session_state.search_submitted = False

# ─── Load Models (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models...")
def load_resources():
    from src.vectorstore import get_embed_model, get_chroma_client, get_collection
    model = get_embed_model()
    client = get_chroma_client()
    coll = get_collection(client)
    return model, client, coll

model, client, coll = load_resources()

# ─── Layout ─────────────────────────────────────────────────────────────────
# The left panel is now in the sidebar (collapsible/resizable like VS Code)
col2, col3 = st.columns([1.2, 1.5])

# ════════════════════════════════════════════════════════════════════════════
# LEFT PANEL (Sidebar) — Stats, Filters, Upload
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.subheader("📊 Store Stats")
    count = coll.count()
    st.success(f"**{count}** resumes indexed")

    st.divider()
    st.subheader("🔧 Filters")
    cat = st.selectbox("Category", ["All"] + sorted(CATEGORY_LIST))
    threshold_pct = st.slider(
        "Min Match %", min_value=0, max_value=100, value=35, step=5,
        help="Semantic similarity threshold. 35-55% is typical for this model."
    )

    new_filters = {"Category": cat, "threshold": threshold_pct / 100.0}

    if st.session_state.search_submitted:
        old_filters = st.session_state.hiring_state.get("filters_applied", {})
        st.session_state.hiring_state["filters_applied"] = new_filters
        # If DB-level filters changed, re-retrieve
        if cat != old_filters.get("Category", "All"):
            from src.graph import retrieve_and_score_node
            with st.spinner("Applying filters..."):
                new_state = retrieve_and_score_node(st.session_state.hiring_state)
                st.session_state.hiring_state.update(new_state)
            st.rerun()
    else:
        st.session_state.hiring_state["filters_applied"] = new_filters

    st.divider()
    upload_container = st.container(border=True)
    with upload_container:
        st.markdown("### 📎 Upload Resumes")
        st.markdown("*Add new PDFs to immediately search against them.*")
        if "uploaded_cache" not in st.session_state:
            st.session_state.uploaded_cache = set()
    
        uploaded_files = st.file_uploader("Upload PDF resumes", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
    if uploaded_files:
        from src.pdf_utils import extract_text
        from src.vectorstore import add_documents
        import time
        new_docs, new_metas, new_ids = [], [], []
        for uf in uploaded_files:
            if uf.name not in st.session_state.uploaded_cache:
                with st.spinner(f"Processing {uf.name}..."):
                    text = extract_text(uf)
                    if text.strip():
                        new_docs.append(f"Role: Unknown\n{text[:3000]}")
                        new_metas.append({
                            "ResumeID": uf.name,
                            "Name": uf.name.replace(".pdf", ""),
                            "Category": "Uploaded",
                            "Source": "uploaded",
                            "Email": "",
                            "Location": ""
                        })
                        new_ids.append(uf.name)
                        st.session_state.uploaded_cache.add(uf.name)
        if new_docs:
            add_documents(new_docs, new_metas, new_ids)
            st.success(f"✅ Added {len(new_docs)} resume(s)")
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# MIDDLE PANEL — JD Input + Results
# ════════════════════════════════════════════════════════════════════════════
with col2:
    if not st.session_state.search_submitted:
        # ── JD Input ──────────────────────────────────────────────────────
        st.header("📝 Job Description")
        job_title = st.text_input("Job Title:", placeholder="e.g. Data Science Fresher")

        tab1, tab2 = st.tabs(["📄 Paste JD (recommended)", "🔲 Structured Form"])
        with tab1:
            jd_desc = st.text_area(
                "Paste your full job description here:",
                height=250,
                placeholder="We are looking for a Data Scientist with..."
            )
        with tab2:
            s_skills = st.text_input("Required Skills:", placeholder="Python, SQL, pandas, scikit-learn")
            s_exp    = st.text_input("Experience:", placeholder="0-2 years / Fresher")
            s_edu    = st.text_input("Education:", placeholder="B.Tech CS / B.Sc Statistics")
            s_other  = st.text_area("Additional Requirements:", placeholder="Any other details...")

        if st.button("🔍 Find Best Candidates", use_container_width=True, type="primary"):
            raw_jd = jd_desc.strip() if jd_desc.strip() else (
                f"Skills required: {s_skills}\nExperience: {s_exp}\nEducation: {s_edu}\n{s_other}"
            )
            if raw_jd.strip():
                with st.spinner("🧠 Analyzing job description and searching candidates..."):
                    from src.graph import app_graph
                    state = st.session_state.hiring_state
                    state["job_title"] = job_title or "Unknown Role"
                    state["job_description_raw"] = raw_jd
                    state["selected_candidate"] = None
                    state["conversation_history"] = []
                    state["feedback_cache"] = {}

                    new_state = app_graph.invoke(state)
                    st.session_state.hiring_state.update(new_state)
                    st.session_state.search_submitted = True
                    st.rerun()
            else:
                st.warning("Please enter a job description first.")
    else:
        # ── Results ───────────────────────────────────────────────────────
        title = st.session_state.hiring_state.get("job_title", "")
        st.subheader(f"Results for: **{title}**")

        all_cands = st.session_state.hiring_state.get("candidates", [])
        thresh = st.session_state.hiring_state.get("filters_applied", {}).get("threshold", 0.35)
        cands = [c for c in all_cands if c["score"] >= thresh]

        col_h1, col_h2 = st.columns([3, 1])
        with col_h1:
            st.write(f"**{len(cands)}** candidates found above **{int(thresh*100)}%** similarity")
        with col_h2:
            if st.button("🔄 New Search"):
                st.session_state.search_submitted = False
                st.session_state.hiring_state["selected_candidate"] = None
                st.rerun()

        if len(cands) == 0:
            st.warning(f"⚠️ No candidates found above {int(thresh*100)}%. Try lowering the threshold slider.")
            st.info(f"ℹ️ Total candidates retrieved: {len(all_cands)}. Best score: {all_cands[0]['score_pct']}%" if all_cands else "No candidates retrieved at all.")
        else:
            st.divider()
            sel_id = (st.session_state.hiring_state.get("selected_candidate") or {}).get("id")
            list_container = st.container(height=650, border=False)
            with list_container:
                for c in cands:
                    score = c["score_pct"]
                    name  = c["metadata"].get("Name", "Unknown")
                    cat_c = c["metadata"].get("Category", "")
                    score_cls = "score-high" if score >= 60 else ("score-med" if score >= 45 else "score-low")

                    with st.container(border=True):
                        ca, cb = st.columns([4, 1])
                        with ca:
                            is_sel = "🎯 " if sel_id == c["id"] else ""
                            st.markdown(f"**{is_sel}{name}**")
                            email = c["metadata"].get("Email", "")
                            # Show email only if it's a real one (not placeholder)
                            if email and email not in ("contact@email.com", "Unknown", ""):
                                st.caption(f"🏷️ {cat_c}  ·  ✉️ {email}")
                            else:
                                st.caption(f"🏷️ {cat_c}")
                        with cb:
                            st.markdown(f'<span class="score-chip {score_cls}">{score}%</span>', unsafe_allow_html=True)

                        if st.button("View Details →", key=f"btn_{c['id']}"):
                            from src.graph import app_graph
                            st.session_state.hiring_state["selected_candidate"] = c
                            with st.spinner("Generating AI feedback..."):
                                new_state = app_graph.invoke(st.session_state.hiring_state)
                                st.session_state.hiring_state.update(new_state)
                            st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — Candidate Details + Chat
# ════════════════════════════════════════════════════════════════════════════
with col3:
    if not st.session_state.search_submitted:
        st.header("💬 AI Assistant")
        st.info("Run a search to start the AI-powered hiring assistant.")
    else:
        selected = st.session_state.hiring_state.get("selected_candidate")

        if selected:
            st.header("👤 Candidate Details")
            details_container = st.container(height=500, border=False)
            with details_container:
                name = selected["metadata"].get("Name", "Unknown")
                cat_c = selected["metadata"].get("Category", "")
                email = selected["metadata"].get("Email", "")
                st.subheader(f"{name}")
                if email and email not in ("contact@email.com", "Unknown", ""):
                    st.caption(f"Category: {cat_c}  ·  ✉️ {email}")
                else:
                    st.caption(f"Category: {cat_c}")

                bd = selected.get("breakdown", {})
                overall = selected["score_pct"]
                comp = bd.get("composite_score", overall)

                st.markdown(f"**Semantic Match:** `{overall}%` | **Contextual Score:** `{comp}%`")

                c1, c2, c3 = st.columns(3)
                c1.metric("Skills", f"{bd.get('skills_score', '-')}%")
                c2.metric("Experience", f"{bd.get('exp_score', '-')}%")
                c3.metric("Education", f"{bd.get('edu_score', '-')}%")

                with st.expander("📄 Resume Preview"):
                    doc = selected.get("document", "")
                    # Skip the "Role: X\n" prefix
                    preview = doc[doc.find('\n')+1:doc.find('\n')+1501] if '\n' in doc else doc[:1500]
                    st.text(preview)

                st.divider()
                st.markdown("### 🤖 AI Feedback")
                import re
                fb = st.session_state.hiring_state.get("feedback_cache", {}).get(selected["id"], "")
                if fb:
                    # Strip any <think>...</think> blocks the LLM may have emitted
                    fb_clean = re.sub(r"<think>.*?</think>", "", fb, flags=re.DOTALL).strip()
                    st.markdown(fb_clean)
                else:
                    st.info("Click 'View Details' on a candidate to generate AI feedback.")

                st.divider()
                st.markdown("**Quick Actions:**")
                cols = st.columns(2)
                for i, sug in enumerate(["Interview Questions", "Skill Gap Analysis", "Validate Score", "Compare with Top 3"]):
                    if cols[i % 2].button(sug, key=f"sug_{i}"):
                        from src.graph import app_graph
                        st.session_state.hiring_state["conversation_history"].append({"role": "user", "content": sug})
                        new_state = app_graph.invoke(st.session_state.hiring_state)
                        st.session_state.hiring_state.update(new_state)
                        st.rerun()
        else:
            st.header("💬 AI Assistant")
            st.write("**Quick Actions:**")
            cols = st.columns(2)
            for i, sug in enumerate(["Show Top 5", "Compare Top 3", "Who is safest hire", "Validate Scores"]):
                if cols[i % 2].button(sug, key=f"asug_{i}"):
                    from src.graph import app_graph
                    st.session_state.hiring_state["conversation_history"].append({"role": "user", "content": sug})
                    new_state = app_graph.invoke(st.session_state.hiring_state)
                    st.session_state.hiring_state.update(new_state)
                    st.rerun()

        # ── Chat ──────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 💬 Chat with AI Recruiter")
        hist = st.session_state.hiring_state.get("conversation_history", [])
        import re as _re
        for msg in hist:
            with st.chat_message(msg["role"]):
                # Always strip <think>...</think> at render time as a safety net
                clean = _re.sub(r"<think>.*?</think>", "", msg["content"], flags=_re.DOTALL).strip()
                st.markdown(clean)

        if user_input := st.chat_input("Ask anything about the candidates..."):
            from src.graph import app_graph
            st.session_state.hiring_state["conversation_history"].append({"role": "user", "content": user_input})
            with st.spinner("Thinking..."):
                new_state = app_graph.invoke(st.session_state.hiring_state)
                st.session_state.hiring_state.update(new_state)
            st.rerun()
