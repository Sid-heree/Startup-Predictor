import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Startup Success Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: #0f1117;
    }

    section[data-testid="stSidebar"] {
        background: #161b27 !important;
        border-right: 1px solid rgba(255,255,255,0.07);
    }
    section[data-testid="stSidebar"] * { color: #d0d0e0 !important; }

    /* Metric cards */
    .metric-card {
        background: #1a1f30;
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #7c8cf8; }
    .metric-label { font-size: 0.8rem; color: #888; margin-top: 0.2rem; letter-spacing: 0.03em; }

    /* Result boxes */
    .success-box {
        background: #0d3b2e;
        border: 1px solid #1a7a5a;
        border-radius: 14px; padding: 1.8rem; text-align: center;
    }
    .fail-box {
        background: #3b0d0d;
        border: 1px solid #7a1a1a;
        border-radius: 14px; padding: 1.8rem; text-align: center;
    }
    .revenue-box {
        background: #2d2400;
        border: 1px solid #7a6000;
        border-radius: 14px; padding: 1.8rem; text-align: center;
    }
    .box-title { font-size: 1rem; font-weight: 600; color: #aaa; letter-spacing: 0.05em; text-transform: uppercase; }
    .box-value { font-size: 2.8rem; font-weight: 800; color: white; margin: 0.3rem 0; }
    .box-sub   { font-size: 0.9rem; color: #888; }

    /* Page heading */
    .page-title {
        font-size: 1.6rem; font-weight: 700; color: white;
        margin-bottom: 0.2rem;
    }
    .page-sub {
        font-size: 0.85rem; color: #666; margin-bottom: 1.5rem;
    }

    /* Button */
    .stButton > button {
        background: #7c8cf8;
        color: white; border: none; border-radius: 10px;
        padding: 0.65rem 1.5rem; font-size: 0.95rem; font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover { background: #6573f0; }

    h1, h2, h3 { color: white !important; }
    p, li { color: #aaa !important; }

    /* Sidebar nav */
    div[data-testid="stRadio"] label { color: #c0c0d0 !important; font-size: 0.95rem; }

    /* Inputs */
    .stSlider label, .stNumberInput label, .stSelectbox label { color: #999 !important; font-size: 0.85rem !important; }
    .stSelectbox > div > div { background: #1a1f30 !important; border-color: rgba(255,255,255,0.12) !important; color: white !important; }
    .stNumberInput input { background: #1a1f30 !important; color: white !important; border-color: rgba(255,255,255,0.12) !important; }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.07) !important; }

    /* Dataframe */
    .stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    log_model   = joblib.load("models/logistic_model.pkl")
    lin_model   = joblib.load("models/linear_model.pkl")
    sc_cls      = joblib.load("models/scaler_cls.pkl")
    sc_reg      = joblib.load("models/scaler_reg.pkl")
    le_industry = joblib.load("models/le_industry.pkl")
    le_stage    = joblib.load("models/le_stage.pkl")
    feat_cls    = joblib.load("models/features_cls.pkl")
    feat_reg    = joblib.load("models/features_reg.pkl")
    return log_model, lin_model, sc_cls, sc_reg, le_industry, le_stage, feat_cls, feat_reg

@st.cache_data
def load_data():
    return pd.read_csv("data/startup_data.csv")

@st.cache_data
def load_metrics():
    try:
        return joblib.load("models/model_metrics.pkl")
    except FileNotFoundError:
        return None

log_model, lin_model, sc_cls, sc_reg, le_industry, le_stage, FEAT_CLS, FEAT_REG = load_models()
df = load_data()
metrics = load_metrics()

INDUSTRIES = sorted(df["industry"].unique().tolist())
STAGES     = sorted(df["funding_stage"].unique().tolist())

# Helper: $M display formatter
def fmt_m(val_in_millions):
    if val_in_millions >= 1:
        return f"${val_in_millions:.1f}M"
    elif val_in_millions >= 0.001:
        return f"${val_in_millions * 1000:.0f}K"
    else:
        return f"${val_in_millions:.2f}M"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡Startup Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "📊 Model Metrics", "🔮 Predict Success", "💰 Revenue Forecast", "📊 Data Explorer"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Models**")
    st.success("✅ Logistic Regression")
    st.success("✅ Linear Regression")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown('<div class="page-title">Startup Success Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">AI-powered predictions using Logistic Regression &amp; Multiple Linear Regression</div>', unsafe_allow_html=True)

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    total     = len(df)
    success_n = int(df["success"].sum())
    success_r = df["success"].mean()
    avg_rev   = df["future_revenue_m"].mean()

    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total}</div><div class="metric-label">Total Startups</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{success_n}</div><div class="metric-label">Successful</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{success_r:.1%}</div><div class="metric-label">Success Rate</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{fmt_m(avg_rev)}</div><div class="metric-label">Avg Future Revenue</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart: success rate by industry
        ind_stats = df.groupby("industry")["success"].mean().reset_index()
        ind_stats.columns = ["Industry", "Success Rate"]
        ind_stats = ind_stats.sort_values("Success Rate", ascending=True)

        fig_bar = px.bar(
            ind_stats,
            x="Success Rate", y="Industry",
            orientation="h",
            title="Success Rate by Industry",
            color="Success Rate",
            color_continuous_scale=[[0, "#3a3a5c"], [1, "#7c8cf8"]],
            text=ind_stats["Success Rate"].apply(lambda x: f"{x:.0%}")
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#aaa",
            title_font_size=14,
            title_font_color="white",
            coloraxis_showscale=False,
            height=360,
            margin=dict(t=40, b=20, l=10, r=10)
        )
        fig_bar.update_traces(textposition="outside", textfont_color="white")
        fig_bar.update_xaxes(showgrid=False, tickformat=".0%", color="#666")
        fig_bar.update_yaxes(showgrid=False, color="#aaa")
        fig_bar.update_layout(transition=dict(duration=0))
        st.plotly_chart(fig_bar, use_container_width=True,
                        config={"displayModeBar": False, "staticPlot": True})

    with col2:
        # Pie chart: successful startups by funding stage
        stage_succ = df[df["success"] == 1].groupby("funding_stage").size().reset_index(name="count")

        fig_pie = px.pie(
            stage_succ,
            names="funding_stage",
            values="count",
            title="Successful Startups by Funding Stage",
            color_discrete_sequence=["#7c8cf8", "#5eead4", "#f97316", "#a78bfa", "#34d399"],
            hole=0.42
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#aaa",
            title_font_size=14,
            title_font_color="white",
            height=360,
            legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#aaa"),
            margin=dict(t=40, b=20, l=10, r=10)
        )
        fig_pie.update_traces(textfont_color="white")
        fig_pie.update_layout(transition=dict(duration=0))
        st.plotly_chart(fig_pie, use_container_width=True,
                        config={"displayModeBar": False, "staticPlot": True})

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Metrics":
    st.markdown('<div class="page-title">📊 Model Performance Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">How accurate are our predictions?</div>', unsafe_allow_html=True)
    
    if metrics:
        # Simple two-column layout for classification metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Classification Model (Success/Fail)")
            
            # Simple metric cards
            st.markdown(f"""
            <div style="background:#1a1f30; border-radius:10px; padding:1rem; margin-bottom:1rem;">
                <div style="font-size:0.85rem; color:#888;">Accuracy</div>
                <div style="font-size:2rem; font-weight:700; color:#7c8cf8;">{metrics['accuracy']:.1%}</div>
            </div>
            <div style="background:#1a1f30; border-radius:10px; padding:1rem; margin-bottom:1rem;">
                <div style="font-size:0.85rem; color:#888;">Precision</div>
                <div style="font-size:2rem; font-weight:700; color:#7c8cf8;">{metrics['precision']:.1%}</div>
            </div>
            <div style="background:#1a1f30; border-radius:10px; padding:1rem;">
                <div style="font-size:0.85rem; color:#888;">Recall</div>
                <div style="font-size:2rem; font-weight:700; color:#7c8cf8;">{metrics['recall']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 💰 Regression Model (Revenue)")
            
            st.markdown(f"""
            <div style="background:#1a1f30; border-radius:10px; padding:1rem; margin-bottom:1rem;">
                <div style="font-size:0.85rem; color:#888;">R² Score</div>
                <div style="font-size:2rem; font-weight:700; color:#7c8cf8;">{metrics['r2']:.3f}</div>
            </div>
            <div style="background:#1a1f30; border-radius:10px; padding:1rem; margin-bottom:1rem;">
                <div style="font-size:0.85rem; color:#888;">RMSE</div>
                <div style="font-size:2rem; font-weight:700; color:#7c8cf8;">${metrics['rmse']:.1f}M</div>
            </div>
            <div style="background:#1a1f30; border-radius:10px; padding:1rem;">
                <div style="font-size:0.85rem; color:#888;">F1 Score</div>
                <div style="font-size:2rem; font-weight:700; color:#7c8cf8;">{metrics['f1']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Simple confusion matrix
        st.markdown("###  Confusion Matrix")
        cm = metrics['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        
        # Simple table for confusion matrix
        st.markdown(f"""
        <div style="background:#1a1f30; border-radius:10px; padding:1rem;">
            <table style="width:100%; text-align:center; border-collapse:collapse;">
                <tr>
                    <th style="padding:0.5rem;"></th>
                    <th style="padding:0.5rem;">Predicted Fail</th>
                    <th style="padding:0.5rem;">Predicted Success</th>
                </tr>
                <tr style="border-top:1px solid #333;">
                    <th style="padding:0.5rem;">Actual Fail</th>
                    <td style="padding:0.5rem; background:#2a2a3a;">{tn}</td>
                    <td style="padding:0.5rem; background:#2a2a3a;">{fp}</td>
                </tr>
                <tr>
                    <th style="padding:0.5rem;">Actual Success</th>
                    <td style="padding:0.5rem; background:#2a2a3a;">{fn}</td>
                    <td style="padding:0.5rem; background:#2a2a3a; color:#4caf50;">{tp}</td>
                </tr>
            </table>
            <div style="margin-top:0.8rem; font-size:0.8rem; color:#888; text-align:center;">
                ✅ Correct predictions: {tp+tn} | ❌ Wrong predictions: {fp+fn}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.warning("⚠️ Model metrics not found. Please run: `python train_model.py`")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT SUCCESS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Success":
    st.markdown('<div class="page-title">🔮 Predict Success</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Logistic Regression · Binary Classification</div>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1.3, 1])

    with col_form:
        c1, c2 = st.columns(2)
        with c1:
            funding     = st.number_input("Funding ($K)", 100.0, 50000.0, 5000.0, 100.0)
            years       = st.slider("Years Active", 0.5, 12.0, 3.0, 0.5)
            competitors = st.slider("# Competitors", 1, 80, 15)
            product_sc  = st.slider("Product Score (1–10)", 1, 10, 7)
            marketing   = st.number_input("Marketing Spend ($K)", 10.0, 10000.0, 500.0, 100.0)
        with c2:
            team_size   = st.slider("Team Size", 2, 120, 25)
            market_sz   = st.number_input("Market Size ($K TAM)", 10000.0, 5000000.0, 500000.0, 10000.0)
            customers   = st.number_input("Customer Count", 10, 10000, 500, 50)
            industry    = st.selectbox("Industry", INDUSTRIES)
            stage       = st.selectbox("Funding Stage", STAGES)

        predict_cls = st.button("🔮 Predict Success / Fail")

    with col_result:
        if predict_cls:
            # Convert K → M for model
            funding_m  = funding / 1000
            market_m   = market_sz / 1000
            mkt_sp_m   = marketing / 1000

            ind_enc   = le_industry.transform([industry])[0]
            stage_enc = le_stage.transform([stage])[0]

            input_row = pd.DataFrame([[
                funding_m, team_size, years, market_m,
                competitors, product_sc, customers,
                mkt_sp_m, ind_enc, stage_enc
            ]], columns=FEAT_CLS)

            input_sc = sc_cls.transform(input_row)
            pred     = log_model.predict(input_sc)[0]
            prob     = log_model.predict_proba(input_sc)[0]

            st.markdown("<br>", unsafe_allow_html=True)

            if pred == 1:
                st.markdown(f"""
                <div class="success-box">
                    <div class="box-title">Result</div>
                    <div class="box-value">✅ SUCCESS</div>
                    <div class="box-sub">Confidence: {prob[1]*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="fail-box">
                    <div class="box-title">Result</div>
                    <div class="box-value">❌ FAIL</div>
                    <div class="box-sub">Confidence: {prob[0]*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge only
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob[1] * 100, 1),
                title={"text": "Success Probability", "font": {"color": "white", "size": 14}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#555"},
                    "bar": {"color": "#7c8cf8"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "bordercolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 40],   "color": "rgba(180,40,40,0.25)"},
                        {"range": [40, 65],  "color": "rgba(200,160,0,0.25)"},
                        {"range": [65, 100], "color": "rgba(40,160,80,0.25)"},
                    ],
                },
                number={"suffix": "%", "font": {"color": "white", "size": 32}}
            ))
            fig_g.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=240,
                margin=dict(t=30, b=10, l=20, r=20),
                transition=dict(duration=0)
            )
            st.plotly_chart(fig_g, use_container_width=True,
                            config={"displayModeBar": False, "staticPlot": True})

        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;border:1px dashed rgba(124,140,248,0.25);
                        border-radius:12px;margin-top:2rem;">
                <div style="font-size:2.5rem">🔮</div>
                <div style="color:#555;margin-top:0.8rem;font-size:0.9rem;">
                    Fill in the inputs and click<br>
                    <span style="color:#7c8cf8;font-weight:600;">Predict Success / Fail</span>
                </div>
            </div>""", unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — REVENUE FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Revenue Forecast":
    st.markdown('<div class="page-title">💰 Revenue Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Multiple Linear Regression · Revenue Prediction</div>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1.3, 1])

    with col_form:
        c1, c2 = st.columns(2)
        with c1:
            r_funding   = st.number_input("Funding ($K)", 100.0, 50000.0, 5000.0, 100.0, key="r_fund")
            r_team      = st.slider("Team Size", 2, 120, 25, key="r_team")
            r_years     = st.slider("Years Active", 0.5, 12.0, 3.0, 0.5, key="r_yr")
            r_market    = st.number_input("Market Size ($K)", 10000.0, 5000000.0, 500000.0, 10000.0, key="r_mkt")
            r_rev1      = st.number_input("Revenue Year 1 ($K)", 10.0, 20000.0, 1000.0, 100.0, key="r_rev1")
        with c2:
            r_burn      = st.number_input("Burn Rate ($K/mo)", 1.0, 5000.0, 300.0, 50.0, key="r_burn")
            r_pscore    = st.slider("Product Score (1–10)", 1, 10, 7, key="r_ps")
            r_customers = st.number_input("Customer Count", 10, 10000, 500, 50, key="r_cust")
            r_mktspend  = st.number_input("Marketing Spend ($K)", 10.0, 10000.0, 500.0, 100.0, key="r_mktsp")
            r_industry  = st.selectbox("Industry", INDUSTRIES, key="r_ind")
            r_stage     = st.selectbox("Funding Stage", STAGES, key="r_stg")

        predict_rev = st.button("💰 Forecast Revenue")

    with col_result:
        if predict_rev:
            # Convert K → M for model
            r_fund_m  = r_funding / 1000
            r_mkt_m   = r_market  / 1000
            r_rev1_m  = r_rev1    / 1000
            r_burn_m  = r_burn    / 1000
            r_mktsp_m = r_mktspend/ 1000

            ind_enc2   = le_industry.transform([r_industry])[0]
            stage_enc2 = le_stage.transform([r_stage])[0]

            input_reg = pd.DataFrame([[
                r_fund_m, r_team, r_years, r_mkt_m,
                r_rev1_m, r_burn_m, r_pscore,
                r_customers, r_mktsp_m, ind_enc2, stage_enc2
            ]], columns=FEAT_REG)

            input_reg_sc = sc_reg.transform(input_reg)
            rev_pred     = lin_model.predict(input_reg_sc)[0]
            rev_pred     = max(rev_pred, 0.001)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="revenue-box">
                <div class="box-title">Projected Revenue</div>
                <div class="box-value">{fmt_m(rev_pred)}</div>
                <div class="box-sub">Estimated Future Revenue</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Percentile comparison
            pct = (df["future_revenue_m"] < rev_pred).mean()
            st.markdown(f"""
            <div style="background:#1a1f30;border-radius:10px;padding:1rem 1.2rem;
                        border:1px solid rgba(255,255,255,0.07);">
                <div style="color:#aaa;font-size:0.85rem;">Percentile vs dataset</div>
                <div style="color:#f9c74f;font-size:1.6rem;font-weight:700;margin-top:0.2rem;">
                    Top {(1-pct)*100:.0f}%
                </div>
                <div style="color:#555;font-size:0.8rem;">
                    Better than {pct:.0%} of startups in dataset
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Simple breakdown card
            st.markdown(f"""
            <div style="background:#1a1f30;border-radius:10px;padding:1rem 1.2rem;
                        border:1px solid rgba(255,255,255,0.07);">
                <div style="color:#aaa;font-size:0.85rem;margin-bottom:0.6rem;">Key Inputs Summary</div>
                <div style="display:flex;justify-content:space-between;margin-bottom:0.4rem;">
                    <span style="color:#666;font-size:0.82rem;">Funding</span>
                    <span style="color:white;font-size:0.82rem;">{fmt_m(r_funding/1000)}</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:0.4rem;">
                    <span style="color:#666;font-size:0.82rem;">Year 1 Revenue</span>
                    <span style="color:white;font-size:0.82rem;">{fmt_m(r_rev1/1000)}</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:0.4rem;">
                    <span style="color:#666;font-size:0.82rem;">Team Size</span>
                    <span style="color:white;font-size:0.82rem;">{r_team} people</span>
                </div>
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:#666;font-size:0.82rem;">Product Score</span>
                    <span style="color:white;font-size:0.82rem;">{r_pscore}/10</span>
                </div>
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;border:1px dashed rgba(249,199,79,0.2);
                        border-radius:12px;margin-top:2rem;">
                <div style="font-size:2.5rem">💰</div>
                <div style="color:#555;margin-top:0.8rem;font-size:0.9rem;">
                    Fill in the inputs and click<br>
                    <span style="color:#f9c74f;font-weight:600;">Forecast Revenue</span>
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.markdown('<div class="page-title">📊 Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Browse all startup records (values in $M)</div>', unsafe_allow_html=True)

    # Filters row
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        outcome_filter  = st.selectbox("Outcome", ["All", "Success", "Fail"])
    with fc2:
        industry_filter = st.selectbox("Industry", ["All"] + INDUSTRIES)
    with fc3:
        stage_filter    = st.selectbox("Funding Stage", ["All"] + STAGES)

    filtered = df.copy()
    if outcome_filter == "Success":      filtered = filtered[filtered["success"] == 1]
    elif outcome_filter == "Fail":       filtered = filtered[filtered["success"] == 0]
    if industry_filter != "All":         filtered = filtered[filtered["industry"] == industry_filter]
    if stage_filter != "All":            filtered = filtered[filtered["funding_stage"] == stage_filter]

    st.markdown(f"<p style='color:#666;font-size:0.82rem;'>{len(filtered)} records</p>", unsafe_allow_html=True)

    # Keep values in $M (no conversion to K)
    display = filtered.copy()
    
    display = display.rename(columns={
        "funding_amount_m":  "Funding ($M)",
        "team_size":         "Team",
        "years_active":      "Years",
        "market_size_m":     "Market ($M)",
        "num_competitors":   "Competitors",
        "revenue_year1_m":   "Rev Y1 ($M)",
        "burn_rate_m":       "Burn ($M/mo)",
        "customer_count":    "Customers",
        "product_score":     "Prod Score",
        "marketing_spend_m": "Mktg ($M)",
        "industry":          "Industry",
        "funding_stage":     "Stage",
        "future_revenue_m":  "Future Rev ($M)",
        "success":           "Success",
    })

    st.dataframe(display, use_container_width=True, height=560)