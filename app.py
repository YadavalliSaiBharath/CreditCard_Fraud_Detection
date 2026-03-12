import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page Config (MUST be first Streamlit call) ──
st.set_page_config(
    page_title="FraudShield — Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════
#  GLOBAL STYLES
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary:    #060B14;
    --bg-secondary:  #0D1526;
    --bg-card:       #111D35;
    --accent-blue:   #00D4FF;
    --accent-green:  #00FF9C;
    --accent-red:    #FF3A5C;
    --accent-orange: #FF8C42;
    --accent-yellow: #FFD600;
    --text-primary:  #E8F0FF;
    --text-secondary:#7A91B8;
    --border:        #1E3050;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}
[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; }

[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0066CC, #00D4FF) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em;
    padding: 0.6rem 1.5rem !important;
    text-transform: uppercase;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 212, 255, 0.35) !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.05em;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent-blue) !important;
    border-bottom-color: var(--accent-blue) !important;
}
hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-blue); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════
def glow_title(text, subtitle=None):
    sub = f'<p style="color:var(--text-secondary);font-size:1rem;margin:0.3rem 0 0 0;font-weight:300;">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div style="margin-bottom:2rem;">
        <h1 style="font-family:Space Mono,monospace;font-size:2rem;font-weight:700;
        color:var(--text-primary);margin:0;letter-spacing:-0.02em;
        text-shadow:0 0 40px rgba(0,212,255,0.4);">🛡️ {text}</h1>
        {sub}
    </div>""", unsafe_allow_html=True)

def section_header(icon, title):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.6rem;margin:1.5rem 0 1rem 0;
    padding-bottom:0.6rem;border-bottom:1px solid var(--border);">
        <span style="font-size:1.1rem;">{icon}</span>
        <span style="font-family:Space Mono,monospace;font-size:0.85rem;
        color:var(--accent-blue);letter-spacing:0.08em;text-transform:uppercase;
        font-weight:700;">{title}</span>
    </div>""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(13,21,38,0.6)',
    font=dict(family='DM Sans', color='#7A91B8', size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor='#1E3050', zerolinecolor='#1E3050'),
    yaxis=dict(gridcolor='#1E3050', zerolinecolor='#1E3050'),
)


# ══════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════
@st.cache_resource
def load_model_and_features():
    try:
        base_dir  = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "fraud_model.pkl")
        pkl_path   = os.path.join(base_dir, "feature_cols.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"fraud_model.pkl not found at: {model_path}")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"feature_cols.pkl not found at: {pkl_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(pkl_path, "rb") as f:
            feature_cols = pickle.load(f)

        return model, feature_cols, None
    except Exception as e:
        return None, None, str(e)


# ══════════════════════════════════════════════
#  SCORING
# ══════════════════════════════════════════════
def score_dataframe(model, feature_cols, pdf):
    amount_mean = pdf['Amount'].mean()
    amount_std  = pdf['Amount'].std() or 1
    time_max    = pdf['Time'].max() or 1
    amount_95th = pdf['Amount'].quantile(0.95)

    pdf['Amount_normalized'] = (pdf['Amount'] - amount_mean) / amount_std
    pdf['Time_normalized']   = pdf['Time'] / time_max
    pdf['hour_of_day']       = (pdf['Time'] / 3600 % 24).astype(int)
    pdf['hour_sin']          = np.sin(2 * np.pi * pdf['hour_of_day'] / 24)
    pdf['hour_cos']          = np.cos(2 * np.pi * pdf['hour_of_day'] / 24)
    pdf['amount_zscore']     = np.abs((pdf['Amount'] - amount_mean) / amount_std)
    pdf['is_high_value']     = (pdf['Amount'] > amount_95th).astype(int)

    available = [c for c in feature_cols if c in pdf.columns]
    X = pdf[available].values

    fraud_prob = model.predict_proba(X)[:, 1]
    prediction = model.predict(X)

    pdf['fraud_probability'] = fraud_prob
    pdf['risk_score']        = (fraud_prob * 100).astype(int)
    pdf['prediction']        = prediction

    def assign_risk(score):
        if score >= 76: return 'CRITICAL'
        if score >= 56: return 'HIGH'
        if score >= 31: return 'MEDIUM'
        return 'LOW'

    def assign_decision(score):
        if score >= 76: return 'BLOCKED'
        if score >= 56: return 'REVIEW'
        if score >= 31: return 'MONITOR'
        return 'APPROVED'

    pdf['risk_level'] = pdf['risk_score'].apply(assign_risk)
    pdf['decision']   = pdf['risk_score'].apply(assign_decision)

    return pdf


# ══════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:1rem 0 1.5rem 0;border-bottom:1px solid var(--border);margin-bottom:1.5rem;">
            <div style="font-family:Space Mono,monospace;font-size:1.1rem;font-weight:700;
            color:var(--accent-blue);letter-spacing:0.05em;">🛡️ FRAUDSHIELD</div>
            <div style="font-size:0.75rem;color:var(--text-secondary);margin-top:0.3rem;">
            Big Data Fraud Detection</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div style="font-family:Space Mono,monospace;font-size:0.7rem;color:var(--text-secondary);letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem;">Navigation</div>', unsafe_allow_html=True)
        page = st.radio("Navigation", ["🏠  Dashboard", "📤  Upload & Analyze", "📊  Analytics", "ℹ️  About"],
                        label_visibility="collapsed")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.72rem;color:var(--text-secondary);line-height:1.8;">
            <div><span style="color:var(--accent-blue);font-family:Space Mono,monospace;">MODEL</span><br>Gradient Boosted Trees</div>
            <div style="margin-top:0.5rem;"><span style="color:var(--accent-blue);font-family:Space Mono,monospace;">ACCURACY</span><br>99.75%</div>
            <div style="margin-top:0.5rem;"><span style="color:var(--accent-blue);font-family:Space Mono,monospace;">ROC-AUC</span><br>0.9993</div>
        </div>""", unsafe_allow_html=True)

    return page


# ══════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════
def page_dashboard():
    glow_title("FraudShield", "Big Data-Driven Credit Card Fraud Detection System")

    cols = st.columns(4)
    for col, (label, val, delta) in zip(cols, [
        ("Transactions Analyzed", "91,200",  "+12.4%"),
        ("Fraud Detection Rate",  "99.38%",  "+0.6%"),
        ("Model Accuracy",        "99.75%",  "GBT"),
        ("ROC-AUC Score",         "0.9993",  "↑ RF"),
    ]):
        with col:
            st.metric(label, val, delta)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("⚙️", "Pipeline Overview")

    steps = [
        ("01", "Data Ingestion",      "PySpark reads CSV into distributed DataFrames",        "#00D4FF"),
        ("02", "Feature Engineering", "Normalize + 6 behavioral features engineered",         "#00FF9C"),
        ("03", "SMOTE Balancing",     "Fraud rate 0.231% → 20% using synthetic samples",      "#FF8C42"),
        ("04", "Model Training",      "Random Forest + GBT via Spark MLlib",                  "#00D4FF"),
        ("05", "Real-Time Scoring",   "Each transaction scored 0–100, 4 risk levels",         "#00FF9C"),
    ]
    cols = st.columns(5)
    for col, (num, title, desc, color) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid var(--border);
            border-top:3px solid {color};border-radius:12px;padding:1.2rem;height:160px;">
                <div style="font-family:Space Mono,monospace;font-size:1.4rem;
                font-weight:700;color:{color};opacity:0.4;">{num}</div>
                <div style="font-weight:600;font-size:0.85rem;
                color:var(--text-primary);margin:0.4rem 0 0.5rem 0;">{title}</div>
                <div style="font-size:0.75rem;color:var(--text-secondary);
                line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("🏆", "Model Performance")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        metrics = ['ROC-AUC', 'Accuracy', 'F1-Score', 'Precision', 'Recall']
        rf_vals  = [0.9999, 0.9944, 0.9944, 0.9944, 0.9944]
        gbt_vals = [0.9993, 0.9975, 0.9975, 0.9975, 0.9975]
        fig.add_trace(go.Bar(name='Random Forest', x=metrics, y=rf_vals,
            marker_color='#00D4FF', opacity=0.8))
        fig.add_trace(go.Bar(name='GBT', x=metrics, y=gbt_vals,
            marker_color='#FF3A5C', opacity=0.8))
        fig.update_layout(**PLOTLY_THEME, title='RF vs GBT — All Metrics',
            yaxis_range=[0.985, 1.002], barmode='group',
            legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        rows = [
            ('ROC-AUC',   '0.9999 🏆', '0.9993'),
            ('Accuracy',  '99.44%',    '99.75% 🏆'),
            ('F1-Score',  '0.9944',    '0.9975 🏆'),
            ('Precision', '0.9944',    '0.9975 🏆'),
            ('Recall',    '0.9944',    '0.9975 🏆'),
        ]
        rows_html = "".join([f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
        padding:0.6rem 0;border-bottom:1px solid var(--border);">
            <span style="font-size:0.85rem;color:var(--text-secondary);">{m}</span>
            <div style="display:flex;gap:0.8rem;">
                <span style="font-family:Space Mono,monospace;font-size:0.8rem;color:#00D4FF;">RF: {rf}</span>
                <span style="font-family:Space Mono,monospace;font-size:0.8rem;color:#FF3A5C;">GBT: {gbt}</span>
            </div>
        </div>""" for m, rf, gbt in rows])

        st.markdown(f"""
        <div style="background:var(--bg-card);border:1px solid var(--border);
        border-radius:14px;padding:1.5rem;height:100%;">
            <div style="font-family:Space Mono,monospace;font-size:0.7rem;
            color:var(--text-secondary);letter-spacing:0.1em;text-transform:uppercase;
            margin-bottom:1rem;">Why GBT Won</div>
            {rows_html}
            <div style="margin-top:1rem;padding:0.8rem;background:rgba(0,255,156,0.05);
            border:1px solid rgba(0,255,156,0.2);border-radius:8px;">
                <span style="font-size:0.8rem;color:var(--accent-green);">
                ✅ GBT selected — wins 4/5 metrics</span>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE: UPLOAD & ANALYZE
# ══════════════════════════════════════════════
def page_upload(model, feature_cols):
    glow_title("Upload & Analyze", "Run fraud detection on your transaction data")

    if model is None:
        st.error("⚠️ Model failed to load. Make sure fraud_model.pkl and feature_cols.pkl are in the same folder as app.py")
        return

    section_header("📤", "Upload Transaction CSV")
    st.markdown('<div style="font-size:0.85rem;color:var(--text-secondary);margin-bottom:1rem;">Upload a CSV with columns: <code style="color:var(--accent-blue);">Time, V1–V28, Amount</code> (optionally <code style="color:var(--accent-blue);">Class</code>)</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop your CSV here", type=['csv'], label_visibility="collapsed")

    if uploaded:
        with st.spinner("Loading data..."):
            pdf = pd.read_csv(uploaded)

        st.success(f"✅ Loaded {len(pdf):,} transactions — {len(pdf.columns)} columns")

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Rows", f"{len(pdf):,}")
        with c2: st.metric("Features", len(pdf.columns))
        with c3:
            if 'Amount' in pdf.columns:
                st.metric("Avg Amount", f"${pdf['Amount'].mean():.2f}")
        with c4:
            if 'Class' in pdf.columns:
                st.metric("Fraud Rate", f"{pdf['Class'].mean()*100:.3f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("👁️", "Data Preview")
        st.dataframe(pdf.head(10), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀  RUN FRAUD DETECTION", use_container_width=True):
            bar = st.progress(0, text="Initializing pipeline...")
            time.sleep(0.3)
            bar.progress(20, text="Engineering features...")
            time.sleep(0.3)
            bar.progress(50, text="Running GBT scoring model...")
            results = score_dataframe(model, feature_cols, pdf.copy())
            bar.progress(80, text="Classifying risk levels...")
            time.sleep(0.2)
            bar.progress(100, text="Done!")
            time.sleep(0.3)
            bar.empty()

            st.session_state['results'] = results
            st.success(f"✅ Scored {len(results):,} transactions!")

    if 'results' in st.session_state:
        results = st.session_state['results']

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("🎯", "Scoring Results")

        total   = len(results)
        blocked = (results['risk_level'] == 'CRITICAL').sum()
        flagged = results['risk_level'].isin(['HIGH', 'CRITICAL']).sum()
        avg_r   = results['risk_score'].mean()

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Scored",    f"{total:,}")
        with c2: st.metric("Blocked",   f"{blocked:,}", f"{blocked/total*100:.1f}%")
        with c3: st.metric("Flagged",   f"{flagged:,}", f"{flagged/total*100:.1f}%")
        with c4: st.metric("Avg Risk",  f"{avg_r:.1f}/100")

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("📋", "Transaction Risk Table")

        display_cols = [c for c in ['Class', 'risk_score', 'risk_level', 'decision', 'Amount_normalized'] if c in results.columns]
        display_df   = results[display_cols].sort_values('risk_score', ascending=False)

        def color_risk(val):
            return {
                'CRITICAL': 'color: #FF3A5C; font-weight: bold',
                'HIGH':     'color: #FF8C42; font-weight: bold',
                'MEDIUM':   'color: #FFD600',
                'LOW':      'color: #00FF9C',
                'BLOCKED':  'color: #FF3A5C; font-weight: bold',
                'REVIEW':   'color: #FF8C42',
                'MONITOR':  'color: #FFD600',
                'APPROVED': 'color: #00FF9C',
            }.get(val, '')

        style_cols = [c for c in ['risk_level', 'decision'] if c in display_df.columns]
        styled = display_df.head(200).style.applymap(color_risk, subset=style_cols)
        if 'risk_score' in display_df.columns:
            styled = styled.background_gradient(subset=['risk_score'], cmap='RdYlGn_r', vmin=0, vmax=100)
        st.dataframe(styled, use_container_width=True, height=400)

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("📈", "Visual Analysis")

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            if 'Class' in results.columns:
                fig.add_trace(go.Histogram(x=results[results['Class']==0]['risk_score'],
                    nbinsx=50, name='Legitimate', marker_color='#00D4FF', opacity=0.6, histnorm='density'))
                fig.add_trace(go.Histogram(x=results[results['Class']==1]['risk_score'],
                    nbinsx=50, name='Fraud', marker_color='#FF3A5C', opacity=0.6, histnorm='density'))
            else:
                fig.add_trace(go.Histogram(x=results['risk_score'], nbinsx=50,
                    marker_color='#00D4FF', opacity=0.8, histnorm='density'))
            fig.add_vline(x=56, line_dash='dash', line_color='#FFD600', annotation_text='Alert Threshold')
            fig.update_layout(**PLOTLY_THEME, title='Risk Score Distribution',
                barmode='overlay', legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            order  = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            colors = ['#00FF9C', '#FFD600', '#FF8C42', '#FF3A5C']
            counts = results['risk_level'].value_counts()
            vals   = [counts.get(r, 0) for r in order]
            fig2 = go.Figure(go.Bar(x=order, y=vals, marker_color=colors,
                text=vals, textposition='outside',
                textfont=dict(color='#E8F0FF', size=13, family='Space Mono')))
            fig2.update_layout(**PLOTLY_THEME, title='Transactions by Risk Level', showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        if 'Class' in results.columns and 'prediction' in results.columns:
            c3, c4 = st.columns(2)
            with c3:
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(results['Class'], results['prediction'])
                fig3 = px.imshow(cm,
                    labels=dict(x='Predicted', y='Actual', color='Count'),
                    x=['Legitimate', 'Fraud'], y=['Legitimate', 'Fraud'],
                    color_continuous_scale=[[0,'#0D1526'],[0.5,'#0066CC'],[1,'#00D4FF']],
                    text_auto=True)
                fig3.update_layout(**PLOTLY_THEME, title='Confusion Matrix')
                fig3.update_traces(textfont=dict(size=16, color='white', family='Space Mono'))
                st.plotly_chart(fig3, use_container_width=True)

            with c4:
                if 'Amount_normalized' in results.columns:
                    sample = results.sample(min(800, len(results)), random_state=42)
                    fig4 = go.Figure()
                    for cls, color, label in [(0,'#00D4FF','Legitimate'),(1,'#FF3A5C','Fraud')]:
                        sub = sample[sample['Class'] == cls]
                        fig4.add_trace(go.Scatter(x=sub['Amount_normalized'], y=sub['risk_score'],
                            mode='markers', name=label,
                            marker=dict(color=color, size=5, opacity=0.6)))
                    fig4.add_hline(y=56, line_dash='dash', line_color='#FFD600')
                    fig4.update_layout(**PLOTLY_THEME, title='Amount vs Risk Score',
                        legend=dict(bgcolor='rgba(0,0,0,0)'),
                        xaxis_title='Normalized Amount', yaxis_title='Risk Score')
                    st.plotly_chart(fig4, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("💾", "Export Results")
        csv_out = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️  Download Scored Results as CSV",
            data=csv_out,
            file_name='fraud_results.csv',
            mime='text/csv',
            use_container_width=True,
            key='download_results'
        )


# ══════════════════════════════════════════════
#  PAGE: ANALYTICS
# ══════════════════════════════════════════════
def page_analytics():
    glow_title("Analytics", "Explore fraud patterns and model behavior")

    section_header("📊", "Risk Level Definitions")
    cols = st.columns(4)
    levels = [
        ("🟢 LOW",      "Score 0–30",   "Approved automatically. Very low fraud probability.",      "#00FF9C"),
        ("🟡 MEDIUM",   "Score 31–55",  "Approved with monitoring. Model is less certain.",         "#FFD600"),
        ("🟠 HIGH",     "Score 56–75",  "Flagged for manual analyst review before processing.",     "#FF8C42"),
        ("🔴 CRITICAL", "Score 76–100", "Blocked immediately. Customer alerted.",                   "#FF3A5C"),
    ]
    for col, (title, score, desc, color) in zip(cols, levels):
        with col:
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid {color}33;
            border-left:4px solid {color};border-radius:10px;padding:1.2rem;height:150px;">
                <div style="font-weight:700;color:{color};margin-bottom:0.3rem;">{title}</div>
                <div style="font-family:Space Mono,monospace;font-size:0.75rem;
                color:var(--text-secondary);margin-bottom:0.5rem;">{score}</div>
                <div style="font-size:0.78rem;color:var(--text-secondary);line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("⚡", "SMOTE Balancing Impact")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(
            x=['Before SMOTE', 'After SMOTE'], y=[0.231, 20.0],
            marker_color=['#FF3A5C', '#00FF9C'],
            text=['0.231%', '20.0%'], textposition='outside',
            textfont=dict(color='#E8F0FF', size=14, family='Space Mono')))
        fig.update_layout(**PLOTLY_THEME, title='Fraud Rate Before vs After SMOTE',
            yaxis_title='Fraud Rate (%)', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = go.Figure(go.Bar(
            x=['Original', 'After SMOTE'], y=[211, 9098],
            marker_color=['#FF3A5C', '#00FF9C'],
            text=['211', '9,098'], textposition='outside',
            textfont=dict(color='#E8F0FF', size=14, family='Space Mono')))
        fig2.update_layout(**PLOTLY_THEME, title='Fraud Sample Count Before vs After',
            yaxis_title='Fraud Cases', showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("🔬", "Engineered Features")
    features = [
        ("Amount_normalized", "Standardized transaction amount (mean=0, std=1)"),
        ("Time_normalized",   "Time scaled to 0–1 range across entire dataset"),
        ("hour_sin",          "Cyclic sine encoding of transaction hour"),
        ("hour_cos",          "Cyclic cosine encoding of transaction hour"),
        ("amount_zscore",     "How many std deviations from mean — anomaly signal"),
        ("is_high_value",     "Binary flag: 1 if amount exceeds 95th percentile"),
    ]
    c1, c2 = st.columns(2)
    for i, (feat, desc) in enumerate(features):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid var(--border);
            border-radius:10px;padding:1rem;margin-bottom:0.6rem;">
                <span style="font-family:Space Mono,monospace;font-size:0.8rem;color:var(--accent-blue);">{feat}</span>
                <span style="font-size:0.78rem;color:var(--text-secondary);margin-left:0.8rem;">{desc}</span>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════
def page_about():
    glow_title("About", "Project details and tech stack")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""
        <div style="background:var(--bg-card);border:1px solid var(--border);
        border-radius:14px;padding:2rem;line-height:1.8;">
            <div style="font-family:Space Mono,monospace;font-size:0.7rem;
            color:var(--text-secondary);letter-spacing:0.1em;text-transform:uppercase;
            margin-bottom:1rem;">Project Summary</div>
            <p style="color:var(--text-secondary);font-size:0.9rem;">
            FraudShield is a Big Data-driven credit card fraud detection system built on
            Apache Spark (PySpark) and Spark MLlib. It processes large-scale transaction
            data using distributed computing and classifies each transaction with a
            real-time fraud risk score between 0 and 100.
            </p>
            <p style="color:var(--text-secondary);font-size:0.9rem;">
            The system addresses key real-world challenges — severe class imbalance
            (0.231% fraud rate), large-scale data processing, and sub-second scoring
            latency. SMOTE combined with random undersampling achieved a balanced
            20% fraud training rate.
            </p>
            <p style="color:var(--text-secondary);font-size:0.9rem;">
            Two Spark MLlib models were trained and compared — Random Forest (100 trees)
            and Gradient Boosted Trees (50 iterations). GBT was selected as the final
            model winning on 4 out of 5 evaluation metrics with 99.75% accuracy.
            </p>
        </div>""", unsafe_allow_html=True)

    with c2:
        tech_stack = [
            ("Apache Spark 4.0", "Distributed processing"),
            ("PySpark",          "Python Spark API"),
            ("Spark MLlib",      "ML at scale"),
            ("SMOTE",            "Class balancing"),
            ("Streamlit",        "Web interface"),
            ("Plotly",           "Interactive charts"),
            ("Kaggle ULB",       "Dataset source"),
        ]
        rows_html = "".join([f"""
        <div style="display:flex;justify-content:space-between;padding:0.5rem 0;
        border-bottom:1px solid var(--border);">
            <span style="font-family:Space Mono,monospace;font-size:0.8rem;color:var(--accent-blue);">{t}</span>
            <span style="font-size:0.78rem;color:var(--text-secondary);">{r}</span>
        </div>""" for t, r in tech_stack])
        st.markdown(f"""
        <div style="background:var(--bg-card);border:1px solid var(--border);
        border-radius:14px;padding:1.5rem;">
            <div style="font-family:Space Mono,monospace;font-size:0.7rem;
            color:var(--text-secondary);letter-spacing:0.1em;text-transform:uppercase;
            margin-bottom:1rem;">Tech Stack</div>
            {rows_html}
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
def main():
    page = render_sidebar()
    model, feature_cols, error         = load_model_and_features()

    with st.sidebar:
        st.markdown("<hr>", unsafe_allow_html=True)
        if error:
            st.markdown(f'<div style="font-size:0.75rem;color:#FF3A5C;">⚠️ Model not loaded<br><span style="opacity:0.7;">{str(error)[:60]}...</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:0.75rem;color:#00FF9C;">✅ Model loaded and ready</div>', unsafe_allow_html=True)

    if   "Dashboard" in page: page_dashboard()
    elif "Upload"    in page: page_upload(model, feature_cols)
    elif "Analytics" in page: page_analytics()
    elif "About"     in page: page_about()


if __name__ == "__main__":
    main()
