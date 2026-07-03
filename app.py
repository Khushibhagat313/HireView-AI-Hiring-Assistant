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
/* Left Panel Redesign */
[data-testid="stSidebar"] { min-width: 300px !important; max-width: 300px !important; }
[data-testid="stSidebar"] hr { display: none; }
[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
    border: none;
    background-color: #fdfdfd;
    border-radius: 8px;
    padding: 5px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-bottom: 15px;
}
[data-testid="stMetricValue"] { color: #4A7C6F; }
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
        total_results=0,
        search_only_candidates=[]
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
    with st.container(border=True):
        st.markdown("### 📊 Store Stats")
        count = coll.count()
        cat_count = len(CATEGORY_LIST) + 1
        c1, c2 = st.columns(2)
        c1.metric("Resumes Indexed", count)
        c2.metric("Categories Covered", cat_count)

    with st.container(border=True):
        st.markdown("### 🎯 Smart Filters")
        cat = st.selectbox("Category", ["All"] + sorted(CATEGORY_LIST))
        threshold_pct = st.slider(
            "Min Match %", min_value=0, max_value=100, value=35, step=5,
            help="Semantic similarity threshold. 35-55% is typical for this model."
        )

    new_filters = {"Category": cat, "threshold": threshold_pct / 100.0}

    if st.session_state.search_submitted:
        old_filters = st.session_state.hiring_state.get("filters_applied", {})
        st.session_state.hiring_state["filters_applied"] = new_filters
        if cat != old_filters.get("Category", "All"):
            from src.graph import retrieve_and_score_node
            with st.spinner("Applying filters..."):
                new_state = retrieve_and_score_node(st.session_state.hiring_state)
                st.session_state.hiring_state.update(new_state)
            st.rerun()
    else:
        st.session_state.hiring_state["filters_applied"] = new_filters

    with st.container(border=True):
        st.markdown("### 📁 Upload Resumes")
        st.markdown("*Add new PDFs to immediately search against them.*")
        if "uploaded_cache" not in st.session_state:
            st.session_state.uploaded_cache = set()
        if "search_only_candidates" not in st.session_state.hiring_state:
            st.session_state.hiring_state["search_only_candidates"] = []
    
        uploaded_file = st.file_uploader("Upload PDF resumes", type="pdf", accept_multiple_files=False, label_visibility="collapsed")
        
        if uploaded_file and uploaded_file.name not in st.session_state.uploaded_cache:
            st.markdown(f"**📄 {uploaded_file.name}**")
            b1, b2 = st.columns(2)
            if b1.button("➕ Add to Store", use_container_width=True):
                from src.pdf_utils import extract_text
                from src.vectorstore import add_documents
                from datetime import datetime
                with st.spinner(f"Adding {uploaded_file.name} to store..."):
                    text = extract_text(uploaded_file)
                    if text.strip():
                        add_documents(
                            documents=[f"Role: Unknown\n{text[:3000]}"],
                            metadatas=[{
                                "ResumeID": uploaded_file.name,
                                "Name": uploaded_file.name.replace(".pdf", ""),
                                "Category": "Uploaded",
                                "Source": "uploaded",
                                "Email": "",
                                "upload_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }],
                            ids=[uploaded_file.name]
                        )
                        st.session_state.uploaded_cache.add(uploaded_file.name)
                        st.success(f"✅ {uploaded_file.name} added to store permanently")
                        st.rerun()
                        
            if b2.button("🔍 Search Only", use_container_width=True):
                from src.pdf_utils import extract_text
                from src.vectorstore import get_embed_model
                with st.spinner(f"Processing {uploaded_file.name} for search..."):
                    text = extract_text(uploaded_file)
                    if text.strip():
                        doc_text = f"Role: Unknown\n{text[:3000]}"
                        model = get_embed_model()
                        # encode returns a numpy array, convert to list
                        emb = model.encode([doc_text])[0].tolist()
                        
                        st.session_state.hiring_state["search_only_candidates"].append({
                            "id": f"temp_{uploaded_file.name}",
                            "document": doc_text,
                            "metadata": {
                                "Name": uploaded_file.name.replace(".pdf", ""),
                                "Category": "Uploaded",
                                "Source": "uploaded",
                                "Email": ""
                            },
                            "embedding": emb,
                            "score": 0.0,
                            "score_pct": 0
                        })
                        st.session_state.uploaded_cache.add(uploaded_file.name)
                        st.success(f"🔍 {uploaded_file.name} will be used for this search only")



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
        st.markdown("""<div style="background-color: #fdfdfd; padding: 25px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); margin-top: 50px;">
<h3 style="color: #3B4F7A; margin-bottom: 20px;">💡 How HireView Works</h3>
<div style="line-height: 1.6; font-size: 14px; margin-bottom: 25px;">
<div style="margin-bottom: 15px;">
<strong style="color: #3B4F7A;">1. 📝 Enter a job description</strong><br>
<span style="color: #666; margin-left: 22px;">Paste or fill the structured form</span>
</div>
<div style="margin-bottom: 15px;">
<strong style="color: #3B4F7A;">2. 🔍 Click Find Best Candidates</strong><br>
<span style="color: #666; margin-left: 22px;">AI matches from 3,500+ resumes</span>
</div>
<div style="margin-bottom: 15px;">
<strong style="color: #3B4F7A;">3. 👤 Click any candidate card</strong><br>
<span style="color: #666; margin-left: 22px;">Get AI feedback and insights</span>
</div>
</div>
<hr style="border: 0; border-top: 1px solid #eaeaea; margin: 20px 0; display: block !important;">
<h4 style="color: #4A7C6F; margin-bottom: 15px;">✨ Things you can ask me:</h4>
<ul style="color: #4A7C6F; font-size: 14px; line-height: 1.8; list-style-type: none; padding-left: 0;">
<li>• "Compare top 3 candidates"</li>
<li>• "Who has the most experience?"</li>
<li>• "Find candidates strong in Python"</li>
<li>• "Validate how scores were calculated"</li>
<li>• "What are this candidate's gaps?"</li>
</ul>
</div>""", unsafe_allow_html=True)
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
