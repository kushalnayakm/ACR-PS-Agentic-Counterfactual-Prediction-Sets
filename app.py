"""
ACR Dashboard - Agentic Counterfactual Reasoning Web Application
A domain-agnostic tool for generating and auditing counterfactual explanations.
Features AUTOMATIC causal rule detection — no manual configuration needed.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acr.engine import ACREngine
from acr.smart_rules import auto_detect_rules, apply_rules
from acr.narrator import get_narrative

# ---- Page Configuration ----
st.set_page_config(
    page_title="ACR Dashboard - Counterfactual Explanations",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
    .main-header p { font-size: 1rem; opacity: 0.9; margin: 0.5rem 0 0 0; }

    .metric-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
        border: 1px solid #e0e4f5;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }
    .metric-card h3 { font-size: 2rem; font-weight: 700; color: #667eea; margin: 0; }
    .metric-card p { font-size: 0.85rem; color: #6b7280; margin: 0.25rem 0 0 0; }

    .rule-immutable {
        background: #fee2e2; border-left: 4px solid #ef4444;
        padding: 0.6rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 0.4rem;
    }
    .rule-constraint {
        background: #fef3c7; border-left: 4px solid #f59e0b;
        padding: 0.6rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 0.4rem;
    }
    .rule-mutable {
        background: #dcfce7; border-left: 4px solid #22c55e;
        padding: 0.6rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 0.4rem;
    }

    .step-box {
        background: #ffffff; border: 2px solid #e5e7eb; border-radius: 12px;
        padding: 1.2rem; margin-bottom: 0.75rem; transition: all 0.3s ease;
    }
    .step-box.active { border-color: #667eea; background: #f8f9ff; box-shadow: 0 4px 16px rgba(102, 126, 234, 0.15); }
    .step-box.done { border-color: #22c55e; background: #f0fdf4; }
    .step-number {
        display: inline-block; width: 28px; height: 28px; border-radius: 50%;
        background: #667eea; color: white; text-align: center; line-height: 28px;
        font-weight: 700; font-size: 0.85rem; margin-right: 0.5rem;
    }

    .result-header {
        font-size: 1.3rem; font-weight: 600; color: #1f2937;
        margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #667eea;
    }

    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%); }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stRadio label, section[data-testid="stSidebar"] p { color: #e0e7ff !important; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---- Session State ----
defaults = {
    'engine': ACREngine(), 'step': 1, 'model_trained': False,
    'cfs_generated': False, 'audit_done': False, 'query_dict': None,
    'raw_cfs': [], 'valid_cfs': [], 'invalid_cfs': [], 'auto_rules': {},
    'narrative': None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---- Header ----
st.markdown("""
<div class="main-header">
    <h1>🔮 ACR Dashboard</h1>
    <p>Agentic Counterfactual Reasoning — Upload any dataset, generate explanations, auto-audit for faithfulness</p>
</div>
""", unsafe_allow_html=True)


# ---- Sidebar ----
with st.sidebar:
    st.markdown("## 🧭 Pipeline Steps")
    steps = [
        ("Upload Dataset", st.session_state.step > 1),
        ("Train Model", st.session_state.model_trained),
        ("Generate & Auto-Audit", st.session_state.audit_done),
    ]
    for i, (label, done) in enumerate(steps, 1):
        status = "done" if done else ("active" if i == st.session_state.step else "")
        icon = "✅" if done else ("🔵" if i == st.session_state.step else "⚪")
        st.markdown(f'<div class="step-box {status}"><span class="step-number">{i}</span> {icon} {label}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧠 How It Works")
    st.markdown("""
    1. **Upload** any dataset (CSV/Excel/JSON)
    2. **Train** a classifier on your data
    3. **Generate** counterfactual suggestions
    4. **Auto-Audit**: The system **automatically** detects which features are immutable (age, race, genetics) and filters impossible suggestions — **no manual setup needed!**
    """)


# ═══════════════════════════════════════
# STEP 1: UPLOAD
# ═══════════════════════════════════════
st.markdown('<div class="result-header">📁 Step 1: Upload Your Dataset</div>', unsafe_allow_html=True)

col_upload, col_preview = st.columns([1, 2])

with col_upload:
    uploaded_file = st.file_uploader("Upload CSV, Excel, or JSON", type=['csv', 'xlsx', 'xls', 'json'], key="uploader")
    st.markdown("**Or try a sample:**")
    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button("🩺 Diabetes", use_container_width=True):
            st.session_state.sample_choice = 'diabetes'
            st.session_state.model_trained = False
            st.session_state.cfs_generated = False
            st.session_state.audit_done = False
    with sc2:
        if st.button("💰 Adult Income", use_container_width=True):
            st.session_state.sample_choice = 'adult'
            st.session_state.model_trained = False
            st.session_state.cfs_generated = False
            st.session_state.audit_done = False

engine = st.session_state.engine
df = None

if uploaded_file:
    try:
        df = engine.load_data(uploaded_file)
        st.session_state.step = max(st.session_state.step, 2)
    except Exception as e:
        st.error(f"Error: {e}")
elif hasattr(st.session_state, 'sample_choice') and st.session_state.sample_choice:
    try:
        if st.session_state.sample_choice == 'diabetes':
            engine.df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
        elif st.session_state.sample_choice == 'adult':
            cols = ['age','workclass','fnlwgt','education','education_num','marital_status',
                    'occupation','relationship','race','sex','capital_gain','capital_loss',
                    'hours_per_week','native_country','income']
            engine.df = pd.read_csv(
                "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv",
                header=None, names=cols, skipinitialspace=True
            ).head(2000)
        engine.df.columns = [c.strip().replace(' ', '_') for c in engine.df.columns]
        df = engine.df
        st.session_state.step = max(st.session_state.step, 2)
    except Exception as e:
        st.error(f"Error loading sample: {e}")

if df is not None:
    with col_preview:
        st.markdown(f"**Loaded:** `{df.shape[0]}` rows × `{df.shape[1]}` columns")
        st.dataframe(df.head(8), use_container_width=True, height=280)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><h3>{df.shape[0]}</h3><p>Total Rows</p></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><h3>{df.shape[1]}</h3><p>Columns</p></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><h3>{len(df.select_dtypes(include=[np.number]).columns)}</h3><p>Numerical</p></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><h3>{len(df.select_dtypes(include=["object"]).columns)}</h3><p>Categorical</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ═══════════════════════════════════════
    # STEP 2: TRAIN MODEL
    # ═══════════════════════════════════════
    st.markdown('<div class="result-header">🎯 Step 2: Configure & Train Model</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        target_col = st.selectbox("🎯 Select Target Feature", options=df.columns.tolist(), index=len(df.columns)-1)
    with c2:
        st.markdown(f"**Target `{target_col}` distribution:**")
        st.write(df[target_col].value_counts().head(10))

    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Training RandomForest classifier..."):
            try:
                engine.detect_features(target_col)
                accuracy = engine.train_model()
                st.session_state.model_trained = True
                st.session_state.cfs_generated = False
                st.session_state.audit_done = False
                st.session_state.step = max(st.session_state.step, 3)
                st.success(f"✅ Model trained! Accuracy: **{accuracy:.1%}**")
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

    # ═══════════════════════════════════════
    # STEP 3: GENERATE + AUTO-AUDIT
    # ═══════════════════════════════════════
    if st.session_state.model_trained:
        st.markdown("---")
        st.markdown('<div class="result-header">🔮 Step 3: Generate Counterfactuals & Auto-Audit</div>', unsafe_allow_html=True)

        # Auto-detect rules and show them
        auto_rules = auto_detect_rules(engine.feature_names, df)
        st.session_state.auto_rules = auto_rules

        st.markdown("#### 🧠 Auto-Detected Causal Rules")
        st.markdown("*The system automatically identified these constraints from your column names:*")

        immutable_feats = [f for f, r in auto_rules.items() if not r['mutable']]
        constrained_feats = [f for f, r in auto_rules.items() if r['mutable'] and r['constraint']]
        mutable_feats = [f for f, r in auto_rules.items() if r['mutable'] and not r['constraint']]

        rule_col1, rule_col2, rule_col3 = st.columns(3)
        with rule_col1:
            st.markdown(f"**🔒 Immutable ({len(immutable_feats)})**")
            for f in immutable_feats:
                st.markdown(f'<div class="rule-immutable">🚫 <strong>{f}</strong><br><small>{auto_rules[f]["reason"]}</small></div>', unsafe_allow_html=True)
            if not immutable_feats:
                st.info("None detected")
        with rule_col2:
            st.markdown(f"**⚠️ Constrained ({len(constrained_feats)})**")
            for f in constrained_feats:
                st.markdown(f'<div class="rule-constraint">⬆️ <strong>{f}</strong><br><small>{auto_rules[f]["reason"]}</small></div>', unsafe_allow_html=True)
            if not constrained_feats:
                st.info("None detected")
        with rule_col3:
            st.markdown(f"**✅ Mutable ({len(mutable_feats)})**")
            for f in mutable_feats:
                st.markdown(f'<div class="rule-mutable">✏️ <strong>{f}</strong><br><small>Can be changed freely</small></div>', unsafe_allow_html=True)

        st.markdown("---")

        # CF generation controls
        cf_c1, cf_c2, cf_c3 = st.columns(3)
        with cf_c1:
            test_samples = engine.get_test_samples(20)
            query_idx = st.selectbox("📋 Select test sample", range(len(test_samples)), format_func=lambda i: f"Sample {i+1}")
        with cf_c2:
            predicted = engine.get_predicted_class(query_idx)
            target_classes = engine.get_target_classes()
            st.markdown(f"**Current prediction:** `{predicted}`")
            desired = st.selectbox("🎯 Desired outcome", options=target_classes, index=0)
        with cf_c3:
            num_cfs = st.slider("Number of counterfactuals", 3, 10, 5)

        st.markdown("**Selected Instance:**")
        st.dataframe(test_samples.iloc[[query_idx]], use_container_width=True)

        if st.button("⚡ Generate & Auto-Audit", type="primary", use_container_width=True):
            with st.spinner("Generating counterfactuals and running causal audit..."):
                try:
                    desired_enc = desired
                    if engine.target in engine.label_encoders:
                        desired_enc = engine.label_encoders[engine.target].transform([str(desired)])[0]

                    query_dict, raw_cfs = engine.generate_counterfactuals(query_idx, desired_enc, num_cfs)
                    st.session_state.query_dict = query_dict
                    st.session_state.raw_cfs = raw_cfs

                    # AUTO-AUDIT using smart rules
                    valid, invalid = apply_rules(query_dict, raw_cfs, auto_rules)
                    st.session_state.valid_cfs = valid
                    st.session_state.invalid_cfs = invalid
                    st.session_state.cfs_generated = True
                    st.session_state.audit_done = True
                    st.session_state.narrative = None  # Reset for fresh LLM call

                    st.success(f"✅ Generated **{len(raw_cfs)}** suggestions → **{len(valid)}** Faithful, **{len(invalid)}** Faithless")
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # ═══════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════
    if st.session_state.audit_done:
        st.markdown("---")

        # ---- LLM Narrative ----
        st.markdown('<div class="result-header">🤖 AI-Generated Explanation (Gemini 2.5 Flash)</div>', unsafe_allow_html=True)
        if not st.session_state.get('narrative'):
            with st.spinner("🧠 Generating AI explanation..."):
                try:
                    narrative = get_narrative(
                        st.session_state.query_dict,
                        st.session_state.valid_cfs,
                        st.session_state.invalid_cfs,
                        engine.feature_names
                    )
                    st.session_state.narrative = narrative
                except Exception as e:
                    st.session_state.narrative = f"Could not generate AI narrative: {e}"
        
        st.info(st.session_state.narrative, icon="🤖")

        st.markdown("---")
        st.markdown('<div class="result-header">📊 Audit Results — Faithful vs Faithless</div>', unsafe_allow_html=True)

        total = len(st.session_state.raw_cfs)
        n_valid = len(st.session_state.valid_cfs)
        n_invalid = len(st.session_state.invalid_cfs)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f'<div class="metric-card"><h3>{total}</h3><p>Total Generated</p></div>', unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div class="metric-card"><h3 style="color:#16a34a">{n_valid}</h3><p>✅ Faithful (Actionable)</p></div>', unsafe_allow_html=True)
        with r3:
            st.markdown(f'<div class="metric-card"><h3 style="color:#dc2626">{n_invalid}</h3><p>❌ Faithless (Discarded)</p></div>', unsafe_allow_html=True)

        # Donut chart
        if total > 0:
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Faithful ✅', 'Faithless ❌'], values=[n_valid, n_invalid],
                marker=dict(colors=['#22c55e', '#ef4444']), hole=0.5,
                textinfo='label+value', textfont=dict(size=14)
            )])
            fig_pie.update_layout(title="Audit Summary", height=350, margin=dict(t=50, b=20, l=20, r=20), font=dict(family="Inter"))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Valid CFs
        if st.session_state.valid_cfs:
            st.markdown("### ✅ Faithful Suggestions (Actionable)")
            for i, cf in enumerate(st.session_state.valid_cfs, 1):
                with st.expander(f"✅ Suggestion #{i}", expanded=(i == 1)):
                    changes = []
                    for feat in engine.feature_names:
                        orig = st.session_state.query_dict.get(feat)
                        new = cf.get(feat)
                        if str(orig) != str(new):
                            try:
                                diff = float(new) - float(orig)
                                direction = "📈" if diff > 0 else "📉"
                                changes.append({"Feature": feat, "Original": orig, "Suggested": new, "Change": f"{direction} {diff:+.2f}"})
                            except (ValueError, TypeError):
                                changes.append({"Feature": feat, "Original": orig, "Suggested": new, "Change": "🔄 Changed"})
                    if changes:
                        st.dataframe(pd.DataFrame(changes), use_container_width=True, hide_index=True)

        # Invalid CFs
        if st.session_state.invalid_cfs:
            st.markdown("### ❌ Faithless Suggestions (Auto-Discarded)")
            for i, item in enumerate(st.session_state.invalid_cfs, 1):
                st.markdown(f"""
                <div class="rule-immutable">
                    <strong>Discarded #{i}:</strong> {item['reason']}
                </div>
                """, unsafe_allow_html=True)

        # Comparison bar chart
        if st.session_state.valid_cfs:
            st.markdown("### 📈 Feature Comparison: Original vs Best Suggestion")
            best_cf = st.session_state.valid_cfs[0]
            num_feats, orig_v, cf_v = [], [], []
            for f in engine.feature_names:
                try:
                    o = float(st.session_state.query_dict.get(f, 0))
                    c = float(best_cf.get(f, 0))
                    num_feats.append(f)
                    orig_v.append(o)
                    cf_v.append(c)
                except (ValueError, TypeError):
                    continue
            if num_feats:
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Original', x=num_feats, y=orig_v, marker_color='#6366f1', text=[f'{v:.1f}' for v in orig_v], textposition='auto'))
                fig.add_trace(go.Bar(name='Counterfactual', x=num_feats, y=cf_v, marker_color='#22c55e', text=[f'{v:.1f}' for v in cf_v], textposition='auto'))
                fig.update_layout(barmode='group', height=400, title="Numerical Feature Changes", font=dict(family="Inter"))
                st.plotly_chart(fig, use_container_width=True)

        # Export
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        export = {
            "original_instance": st.session_state.query_dict,
            "auto_detected_rules": {f: {"mutable": r["mutable"], "constraint": r["constraint"], "reason": r["reason"]} for f, r in st.session_state.auto_rules.items()},
            "total_generated": total, "faithful_count": n_valid, "faithless_count": n_invalid,
            "faithful_suggestions": st.session_state.valid_cfs,
            "faithless_suggestions": st.session_state.invalid_cfs,
        }
        st.download_button("📥 Download Full Audit Report (JSON)", json.dumps(export, indent=4, default=str), "acr_audit_report.json", "application/json", use_container_width=True)

# Footer
st.markdown("""
<div style="text-align:center; color:#9ca3af; font-size:0.85rem; padding:1rem;">
    <strong>ACR Dashboard</strong> — Agentic Counterfactual Reasoning |
    Built for XAI Project (6th Semester) |
    Powered by DiCE, Scikit-Learn & Streamlit
</div>
""", unsafe_allow_html=True)
