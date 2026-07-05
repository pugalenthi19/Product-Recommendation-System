# -*- coding: utf-8 -*-
"""
Electronics Sentiment Analysis App
=====================================
Positive vs Negative Review Analysis
Dataset: Enhanced_Dataset_Final.csv
Run: streamlit run electronics_sentiment_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Electronics Sentiment Analyser",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    body { background-color: #0f0f1a; }
    .main { background-color: #0f0f1a; }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        border: 1px solid #2a2a4a;
    }
    .app-title {
        font-size: 32px; font-weight: 900;
        color: #e2e8f0; margin: 0;
        letter-spacing: -0.5px;
    }
    .app-sub { color: #94a3b8; font-size: 15px; margin-top: 6px; }

    /* Category badge */
    .cat-badge {
        display: inline-block;
        background: #1e3a5f;
        color: #60a5fa;
        border: 1px solid #2a5a9f;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e1e2e, #252540);
        border-radius: 14px;
        padding: 20px 18px;
        text-align: center;
        border: 1px solid #2a2a4a;
        height: 100%;
    }
    .kpi-label { color: #94a3b8; font-size: 12px; font-weight: 700;
                 text-transform: uppercase; letter-spacing: 1px; }
    .kpi-value { font-size: 36px; font-weight: 900; margin: 8px 0 4px; }
    .kpi-sub   { color: #64748b; font-size: 12px; }

    /* Review cards */
    .pos-card {
        background: #0d2318;
        border-left: 5px solid #22c55e;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
    }
    .neg-card {
        background: #2a0d0d;
        border-left: 5px solid #ef4444;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
    }
    .review-text { color: #e2e8f0; font-size: 14px; line-height: 1.6; }
    .review-meta { color: #64748b; font-size: 12px; margin-top: 8px; }
    .score-pos   { color: #22c55e; font-weight: 700; }
    .score-neg   { color: #ef4444; font-weight: 700; }

    /* Product info card */
    .product-info {
        background: linear-gradient(135deg, #1e1e2e, #252540);
        border-radius: 14px;
        padding: 20px 24px;
        border: 1px solid #2a2a4a;
        margin: 12px 0;
    }
    .product-name { color: #e2e8f0; font-size: 18px; font-weight: 800; }
    .spec-row { display: flex; gap: 24px; flex-wrap: wrap; margin-top: 10px; }
    .spec-item { color: #94a3b8; font-size: 13px; }
    .spec-val  { color: #cbd5e1; font-weight: 600; }

    /* Section headers */
    .section-pos {
        background: linear-gradient(90deg, #0d2318, #1e1e2e);
        border-radius: 10px; padding: 12px 20px;
        border-left: 5px solid #22c55e;
        margin: 16px 0 10px;
        color: #22c55e; font-size: 18px; font-weight: 800;
    }
    .section-neg {
        background: linear-gradient(90deg, #2a0d0d, #1e1e2e);
        border-radius: 10px; padding: 12px 20px;
        border-left: 5px solid #ef4444;
        margin: 16px 0 10px;
        color: #ef4444; font-size: 18px; font-weight: 800;
    }

    /* Star badge */
    .star-pos { background: #14532d; color: #22c55e; padding: 2px 9px;
                border-radius: 20px; font-size: 11px; font-weight: 700; }
    .star-neg { background: #450a0a; color: #ef4444; padding: 2px 9px;
                border-radius: 20px; font-size: 11px; font-weight: 700; }
    .star-neu { background: #431407; color: #f97316; padding: 2px 9px;
                border-radius: 20px; font-size: 11px; font-weight: 700; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #0f0f1a; border-right: 1px solid #1e1e2e; }
    section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    .stSelectbox label, .stMultiSelect label, .stSlider label,
    .stTextInput label, .stRadio label { color: #94a3b8 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        background: #1e1e2e; border-radius: 8px 8px 0 0;
        color: #94a3b8; font-weight: 600; padding: 0 20px; height: 44px;
    }
    .stTabs [aria-selected="true"] {
        background: #3b82f6 !important; color: #fff !important;
    }

    hr { border-color: #2a2a4a !important; }
    h1,h2,h3,h4 { color: #e2e8f0 !important; }
    p, li { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CATEGORY MAPPING  (from form_factor column)
# ─────────────────────────────────────────────
FORM_FACTOR_TO_CATEGORY = {
    "bar":                   "Smartphone",
    "touchscreen phone":     "Smartphone",
    "smartphone":            "Smartphone",
    "dual sim":              "Smartphone",
    "tablet":                "Tablet",
    "laptop":                "Laptop",
    "portable speaker":      "Bluetooth Speaker",
    "in-ear":                "TWS Earbuds",
    "neckband":              "Neckband",
    "in-ear wired":          "Wired Earphone",
    "smartwatch":            "Smartwatch",
    "power bank":            "Power Bank",
    "adapter":               "Charger & Cable",
    "gaming peripheral":     "Gaming Accessory",
    "console":               "Gaming Console",
    "flat panel monitor":    "Monitor",
    "flat panel":            "Monitor",
    "keyboard/mouse":        "Keyboard & Mouse",
    "desktop printer":       "Printer & Scanner",
    "flatbed scanner":       "Printer & Scanner",
    "webcam":                "Webcam",
    "smart tv":              "Smart TV",
    "streaming stick":       "Streaming Device",
    "smart plug":            "Smart Plug",
    "ip camera":             "CCTV Camera",
    "storage drive":         "SSD & HDD",
    "ram module":            "RAM",
    "pcie graphics card":    "Graphics Card",
    "atx motherboard":       "Motherboard",
    "atx / matx":            "Motherboard",
    "m.2 / 2.5 inch ssd":   "SSD & HDD",
    "3.5 inch hdd":          "SSD & HDD",
    "dimm / so-dimm":        "RAM",
}

CATEGORY_ICON = {
    "Smartphone":       "📱",
    "Tablet":           "📟",
    "Laptop":           "💻",
    "Bluetooth Speaker":"🔊",
    "TWS Earbuds":      "🎧",
    "Neckband":         "🎵",
    "Wired Earphone":   "🎙️",
    "Smartwatch":       "⌚",
    "Power Bank":       "🔋",
    "Charger & Cable":  "🔌",
    "Gaming Accessory": "🎮",
    "Gaming Console":   "🕹️",
    "Monitor":          "🖥️",
    "Keyboard & Mouse": "⌨️",
    "Printer & Scanner":"🖨️",
    "Webcam":           "📷",
    "Smart TV":         "📺",
    "Streaming Device": "📡",
    "Smart Plug":       "🔌",
    "CCTV Camera":      "📹",
    "SSD & HDD":        "💾",
    "RAM":              "🧠",
    "Graphics Card":    "🎴",
    "Motherboard":      "🔧",
    "Other":            "⚙️",
}

def map_category(form_factor: str) -> str:
    if pd.isna(form_factor):
        return "Other"
    key = str(form_factor).strip().lower()
    return FORM_FACTOR_TO_CATEGORY.get(key, "Other")


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Enhanced_Dataset_Final_1.csv")

    # Derived columns
    df["price_clean"] = (
        df["price"].astype(str)
        .str.replace("[^0-9.]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )
    df["rating_clean"] = (
        df["rating"].astype(str)
        .str.extract(r"([\d.]+)")[0]
        .pipe(pd.to_numeric, errors="coerce")
    )
    df["reviews_count"] = (
        df["total_reviews"].astype(str)
        .str.replace("[^0-9]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )
    df["ram_gb"] = (
        df["ram"].astype(str)
        .str.extract(r"(\d+)")[0]
        .pipe(pd.to_numeric, errors="coerce")
    )
    df["category"] = df["form_factor"].apply(map_category)

    # Flatten reviews
    rows = []
    for _, row in df.iterrows():
        reviews = str(row["reviews"]).split("||")
        ratings = str(row["reviews_rating"]).split("||")
        for rv, rt in zip(reviews, ratings):
            rv = rv.strip()
            if rv and rv.lower() not in ("nan", ""):
                score_match = re.search(r"([\d.]+)", rt)
                rows.append({
                    "product_id":   row["Unnamed: 0"],
                    "title":        row["title"],
                    "manufacturer": row["manufacturer"],
                    "category":     row["category"],
                    "price_clean":  row["price_clean"],
                    "rating_clean": row["rating_clean"],
                    "os":           row["os"],
                    "ram_gb":       row["ram_gb"],
                    "colour":       row["colour"],
                    "review_text":  rv,
                    "star_rating":  float(score_match.group(1)) if score_match else np.nan,
                })
    reviews_df = pd.DataFrame(rows)
    return df, reviews_df


@st.cache_resource
def get_analyzer():
    return SentimentIntensityAnalyzer()


@st.cache_data
def run_sentiment(_reviews_df):
    analyzer = get_analyzer()
    results = [analyzer.polarity_scores(str(t)) for t in _reviews_df["review_text"]]
    scores_df = pd.DataFrame(results)
    out = _reviews_df.copy()
    out["compound"]  = scores_df["compound"].values
    out["pos_score"] = scores_df["pos"].values
    out["neg_score"] = scores_df["neg"].values
    out["neu_score"] = scores_df["neu"].values
    out["sentiment"] = out["compound"].apply(
        lambda c: "Positive" if c >= 0.05 else ("Negative" if c <= -0.05 else "Neutral")
    )
    return out


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("Loading dataset and running sentiment analysis…"):
    df, reviews_df = load_data()
    rev = run_sentiment(reviews_df)

BRANDS       = sorted(df["manufacturer"].dropna().unique().tolist())
CATEGORIES   = sorted(df["category"].dropna().unique().tolist())
TOTAL_REVIEWS = len(rev)
POS_COUNT = (rev["sentiment"] == "Positive").sum()
NEG_COUNT = (rev["sentiment"] == "Negative").sum()
NEU_COUNT = (rev["sentiment"] == "Neutral").sum()


# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔌 Filters")
    st.markdown("---")

    st.markdown("**Product Category**")
    all_cats = ["All Categories"] + CATEGORIES
    sel_category_filter = st.selectbox(
        "Filter by category",
        all_cats,
        label_visibility="collapsed",
    )

    st.markdown("**Brand**")
    if sel_category_filter != "All Categories":
        cat_brands = sorted(
            df[df["category"] == sel_category_filter]["manufacturer"]
            .dropna().unique().tolist()
        )
    else:
        cat_brands = BRANDS
    sel_brand_filter = st.selectbox(
        "Filter by brand",
        ["All Brands"] + cat_brands,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Price Range (₹)**")
    min_p = int(df["price_clean"].dropna().min())
    max_p = int(df["price_clean"].dropna().max())
    price_range = st.slider(
        "Price range",
        min_value=min_p, max_value=max_p,
        value=(min_p, max_p),
        step=500,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Min Star Rating**")
    min_star = st.slider(
        "Min star rating",
        0.0, 5.0, 0.0, 0.5,
        label_visibility="collapsed",
    )

    st.markdown("---")
    # Stats summary in sidebar
    st.markdown("### 📊 Dataset Overview")
    st.markdown(f"- **Products:** {len(df):,}")
    st.markdown(f"- **Brands:** {df['manufacturer'].nunique()}")
    st.markdown(f"- **Categories:** {df['category'].nunique()}")
    st.markdown(f"- **Total Reviews:** {TOTAL_REVIEWS:,}")


# Apply sidebar filters to a working subset
def apply_filters(data_df, data_rev):
    d = data_df.copy()
    r = data_rev.copy()
    if sel_category_filter != "All Categories":
        d = d[d["category"] == sel_category_filter]
        r = r[r["category"] == sel_category_filter]
    if sel_brand_filter != "All Brands":
        d = d[d["manufacturer"] == sel_brand_filter]
        r = r[r["manufacturer"] == sel_brand_filter]
    d = d[d["price_clean"].between(price_range[0], price_range[1], inclusive="both") |
          d["price_clean"].isna()]
    if min_star > 0:
        d = d[d["rating_clean"] >= min_star]
        r = r[r["rating_clean"] >= min_star]
    valid_ids = set(d["Unnamed: 0"].tolist())
    r = r[r["product_id"].isin(valid_ids)]
    return d, r

fdf, frev = apply_filters(df, rev)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
active_filter_str = (
    f"{sel_category_filter}" if sel_category_filter != "All Categories"
    else "All Electronics"
)
if sel_brand_filter != "All Brands":
    active_filter_str += f" · {sel_brand_filter}"

cat_icon = CATEGORY_ICON.get(sel_category_filter, "🔌")

st.markdown(f"""
<div class="app-header">
  <div class="app-title">{cat_icon} Electronics Sentiment Analyser</div>
  <div class="app-sub">
    Positive &amp; Negative Review Analysis &nbsp;|&nbsp;
    {len(fdf):,} Products &nbsp;|&nbsp;
    {fdf['manufacturer'].nunique()} Brands &nbsp;|&nbsp;
    {fdf['category'].nunique()} Categories &nbsp;|&nbsp;
    VADER NLP Engine &nbsp;|&nbsp;
    <b style="color:#60a5fa;">Viewing: {active_filter_str}</b>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# GLOBAL KPI STRIP  (filtered)
# ─────────────────────────────────────────────
f_pos = (frev["sentiment"] == "Positive").sum()
f_neg = (frev["sentiment"] == "Negative").sum()
f_neu = (frev["sentiment"] == "Neutral").sum()
f_total = len(frev)
safe_total = f_total if f_total > 0 else 1

k1, k2, k3, k4, k5 = st.columns(5)
k1.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Products (Filtered)</div>
  <div class="kpi-value" style="color:#60a5fa;">{len(fdf):,}</div>
  <div class="kpi-sub">{fdf['category'].nunique()} categories</div>
</div>""", unsafe_allow_html=True)

k2.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Total Reviews</div>
  <div class="kpi-value" style="color:#60a5fa;">{f_total:,}</div>
  <div class="kpi-sub">analysed by VADER</div>
</div>""", unsafe_allow_html=True)

k3.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Positive Reviews</div>
  <div class="kpi-value" style="color:#22c55e;">{f_pos:,}</div>
  <div class="kpi-sub">{f_pos/safe_total*100:.1f}% of all reviews</div>
</div>""", unsafe_allow_html=True)

k4.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Negative Reviews</div>
  <div class="kpi-value" style="color:#ef4444;">{f_neg:,}</div>
  <div class="kpi-sub">{f_neg/safe_total*100:.1f}% of all reviews</div>
</div>""", unsafe_allow_html=True)

k5.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Avg. VADER Score</div>
  <div class="kpi-value" style="color:#a78bfa;">{frev['compound'].mean() if f_total > 0 else 0.0:+.3f}</div>
  <div class="kpi-sub">-1.0 (negative) → +1.0 (positive)</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  🔍  Search by Product  ",
    "  🏷️  Search by Brand  ",
    "  📂  Browse by Category  ",
    "  📊  Overall Analytics  ",
])


# ══════════════════════════════════════════════
# TAB 1 — SEARCH BY PRODUCT
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### 🔍 Search Any Electronic Product")

    search_input = st.text_input(
        "Type a product name, brand, or model",
        placeholder="e.g. Samsung Galaxy M42, Razer Blade 14, boAt Airdopes, Logitech MX Keys…",
    )

    if search_input.strip():
        matches = fdf[fdf["title"].str.contains(search_input.strip(), case=False, na=False)]

        if matches.empty:
            st.warning("No products found. Try a different search term or adjust sidebar filters.")
        else:
            chosen_title = st.selectbox(
                f"{len(matches)} product(s) found — select one:",
                matches["title"].tolist(),
            )

            chosen_full = fdf[fdf["title"] == chosen_title].iloc[0]
            pid = chosen_full["Unnamed: 0"]
            pcat = chosen_full["category"]
            picon = CATEGORY_ICON.get(pcat, "⚙️")

            _price  = chosen_full["price_clean"]
            _rating = chosen_full["rating_clean"]
            _ram    = chosen_full["ram_gb"]
            _stock  = str(chosen_full["availability_status"])
            _mfr    = str(chosen_full["manufacturer"])
            _os     = str(chosen_full["os"])
            _clr    = str(chosen_full["colour"])
            _sf     = str(chosen_full["special_features"])
            _rev_ct = str(chosen_full["total_reviews"])
            _ff     = str(chosen_full["form_factor"])
            _wt     = str(chosen_full.get("weight", "N/A"))
            _bp     = chosen_full.get("battery_power_rating", None)

            price_disp  = ("₹" + f"{int(_price):,}") if pd.notna(_price) else "N/A"
            rating_disp = (str(round(float(_rating), 1)) + " / 5") if pd.notna(_rating) else "N/A"
            ram_disp    = (str(int(_ram)) + " GB") if pd.notna(_ram) else "N/A"
            bp_disp     = (str(int(_bp)) + " mAh") if pd.notna(_bp) and int(_bp) > 0 else "N/A"

            st.markdown(f"""<div class="cat-badge">{picon} {pcat}</div>""", unsafe_allow_html=True)
            st.markdown("""<div class="product-info">""", unsafe_allow_html=True)
            st.markdown(
                f"<div class='product-name'>{chosen_title}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"""
<div class="spec-row" style="margin-top:14px;">
  <div class="spec-item">💰 Price &nbsp;<span class="spec-val">{price_disp}</span></div>
  <div class="spec-item">⭐ Rating &nbsp;<span class="spec-val">{rating_disp}</span></div>
  <div class="spec-item">🏭 Brand &nbsp;<span class="spec-val">{_mfr}</span></div>
  <div class="spec-item">📂 Category &nbsp;<span class="spec-val">{pcat}</span></div>
  <div class="spec-item">🖥️ OS &nbsp;<span class="spec-val">{_os[:30]}</span></div>
  <div class="spec-item">🧠 RAM &nbsp;<span class="spec-val">{ram_disp}</span></div>
  <div class="spec-item">🎨 Colour &nbsp;<span class="spec-val">{_clr}</span></div>
  <div class="spec-item">📦 Stock &nbsp;<span class="spec-val">{_stock[:30]}</span></div>
  <div class="spec-item">💬 Reviews &nbsp;<span class="spec-val">{_rev_ct}</span></div>
  <div class="spec-item">🔋 Battery &nbsp;<span class="spec-val">{bp_disp}</span></div>
  <div class="spec-item">⚖️ Weight &nbsp;<span class="spec-val">{_wt}</span></div>
</div>
""", unsafe_allow_html=True)
            if _sf and _sf not in ("nan", "N/A"):
                st.markdown(
                    "<div style='margin-top:10px; color:#94a3b8; font-size:13px;'>"
                    "<b style='color:#cbd5e1;'>Key Features:</b> " + _sf[:250] + "</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

            # Pull reviews for this product
            prod_rev = rev[rev["product_id"] == pid].copy()

            if prod_rev.empty:
                st.info("No reviews available for this product.")
            else:
                pos_r = prod_rev[prod_rev["sentiment"] == "Positive"].sort_values("compound", ascending=False)
                neg_r = prod_rev[prod_rev["sentiment"] == "Negative"].sort_values("compound", ascending=True)
                neu_r = prod_rev[prod_rev["sentiment"] == "Neutral"]
                p_cnt, n_cnt, u_cnt, t_cnt = len(pos_r), len(neg_r), len(neu_r), len(prod_rev)

                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                m1.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Total Reviews</div>
  <div class="kpi-value" style="color:#60a5fa;">{t_cnt}</div>
</div>""", unsafe_allow_html=True)
                m2.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Positive</div>
  <div class="kpi-value" style="color:#22c55e;">{p_cnt}</div>
  <div class="kpi-sub">{p_cnt/t_cnt*100:.0f}%</div>
</div>""", unsafe_allow_html=True)
                m3.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Negative</div>
  <div class="kpi-value" style="color:#ef4444;">{n_cnt}</div>
  <div class="kpi-sub">{n_cnt/t_cnt*100:.0f}%</div>
</div>""", unsafe_allow_html=True)
                m4.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Avg VADER</div>
  <div class="kpi-value" style="color:#a78bfa;">{prod_rev['compound'].mean():+.3f}</div>
</div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                col_chart, col_gauge = st.columns(2)

                with col_chart:
                    donut = px.pie(
                        values=[p_cnt, u_cnt, n_cnt],
                        names=["Positive", "Neutral", "Negative"],
                        hole=0.6,
                        color_discrete_map={"Positive":"#22c55e","Neutral":"#f97316","Negative":"#ef4444"},
                        title="Sentiment Breakdown",
                    )
                    donut.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e2e8f0", height=280,
                        margin=dict(t=40,b=10,l=10,r=10), title_font_size=14,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                    )
                    st.plotly_chart(donut, use_container_width=True)

                with col_gauge:
                    avg_c = round(prod_rev["compound"].mean(), 3)
                    gcol  = "#22c55e" if avg_c >= 0.05 else "#ef4444" if avg_c <= -0.05 else "#f97316"
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=avg_c,
                        gauge={
                            "axis": {"range": [-1, 1], "tickcolor": "#94a3b8"},
                            "bar":  {"color": gcol},
                            "bgcolor": "#1e1e2e",
                            "steps": [
                                {"range": [-1, -0.05], "color": "#2a0d0d"},
                                {"range": [-0.05, 0.05], "color": "#1e1e2e"},
                                {"range": [0.05, 1],  "color": "#0d2318"},
                            ],
                        },
                        title={"text":"Overall Sentiment Score","font":{"color":"#e2e8f0","size":14}},
                        number={"font":{"color":gcol,"size":36}},
                    ))
                    gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                        height=280, margin=dict(t=60,b=10,l=20,r=20),
                    )
                    st.plotly_chart(gauge, use_container_width=True)

                st.markdown("<hr>", unsafe_allow_html=True)

                # Positive reviews
                st.markdown(
                    f"<div class='section-pos'>✅ Positive Reviews — {p_cnt}</div>",
                    unsafe_allow_html=True,
                )
                if pos_r.empty:
                    st.info("No positive reviews found.")
                else:
                    max_p = min(p_cnt, 40)
                    show_p = st.slider("Show top N positive reviews", 1, max_p, min(p_cnt, 5), key="pos_slider_p1") if max_p > 1 else max_p
                    for _, row in pos_r.head(show_p).iterrows():
                        sv = row["star_rating"]
                        ss = (str(round(sv,1)) + " / 5") if pd.notna(sv) else "N/A"
                        cv = round(row["compound"], 3)
                        st.markdown(f"""
<div class="pos-card">
  <div class="review-text">{row['review_text']}</div>
  <div class="review-meta">
    <span class="star-pos">★ {ss}</span> &nbsp;&nbsp;
    VADER Score: <span class="score-pos">{cv:+.3f}</span> &nbsp;&nbsp;
    Positive weight: <span class="score-pos">{round(row['pos_score'],2)}</span>
  </div>
</div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)

                # Negative reviews
                st.markdown(
                    f"<div class='section-neg'>❌ Negative Reviews — {n_cnt}</div>",
                    unsafe_allow_html=True,
                )
                if neg_r.empty:
                    st.info("No negative reviews found.")
                else:
                    max_n = min(n_cnt, 40)
                    show_n = st.slider("Show top N negative reviews", 1, max_n, min(n_cnt, 5), key="neg_slider_p1") if max_n > 1 else max_n
                    for _, row in neg_r.head(show_n).iterrows():
                        sv = row["star_rating"]
                        ss = (str(round(sv,1)) + " / 5") if pd.notna(sv) else "N/A"
                        cv = round(row["compound"], 3)
                        st.markdown(f"""
<div class="neg-card">
  <div class="review-text">{row['review_text']}</div>
  <div class="review-meta">
    <span class="star-neg">★ {ss}</span> &nbsp;&nbsp;
    VADER Score: <span class="score-neg">{cv:+.3f}</span> &nbsp;&nbsp;
    Negative weight: <span class="score-neg">{round(row['neg_score'],2)}</span>
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — SEARCH BY BRAND
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### 🏷️ Brand-Level Sentiment Analysis")

    avail_brands = sorted(fdf["manufacturer"].dropna().unique().tolist())
    sel_brand = st.selectbox("Select a brand", avail_brands, key="brand_tab2")

    brand_rev_t2  = frev[frev["manufacturer"] == sel_brand].copy()
    brand_prods   = fdf[fdf["manufacturer"] == sel_brand]

    # Category chips for this brand
    brand_cats = brand_prods["category"].value_counts()
    chips_html = " ".join([
        f"<span style='background:#1e3a5f;color:#60a5fa;border-radius:20px;"
        f"padding:3px 11px;font-size:11px;font-weight:700;margin:2px;display:inline-block;'>"
        f"{CATEGORY_ICON.get(c,'⚙️')} {c} ({n})</span>"
        for c, n in brand_cats.items()
    ])
    st.markdown(f"<div style='margin:8px 0 16px;'>{chips_html}</div>", unsafe_allow_html=True)

    b_total = len(brand_rev_t2)
    b_pos   = (brand_rev_t2["sentiment"] == "Positive").sum()
    b_neg   = (brand_rev_t2["sentiment"] == "Negative").sum()
    b_neu   = (brand_rev_t2["sentiment"] == "Neutral").sum()
    b_safe  = b_total if b_total > 0 else 1

    b1, b2, b3, b4, b5 = st.columns(5)
    b1.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Products</div>
  <div class="kpi-value" style="color:#60a5fa;">{len(brand_prods)}</div>
</div>""", unsafe_allow_html=True)
    b2.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Total Reviews</div>
  <div class="kpi-value" style="color:#60a5fa;">{b_total:,}</div>
</div>""", unsafe_allow_html=True)
    b3.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Positive</div>
  <div class="kpi-value" style="color:#22c55e;">{b_pos:,}</div>
  <div class="kpi-sub">{b_pos/b_safe*100:.1f}%</div>
</div>""", unsafe_allow_html=True)
    b4.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Negative</div>
  <div class="kpi-value" style="color:#ef4444;">{b_neg:,}</div>
  <div class="kpi-sub">{b_neg/b_safe*100:.1f}%</div>
</div>""", unsafe_allow_html=True)
    b5.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Avg VADER</div>
  <div class="kpi-value" style="color:#a78bfa;">{brand_rev_t2['compound'].mean() if b_total > 0 else 0.0:+.3f}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)

    with ch1:
        bd = px.pie(
            values=[b_pos, b_neu, b_neg],
            names=["Positive", "Neutral", "Negative"],
            hole=0.55,
            title=f"{sel_brand} — Sentiment Split",
            color_discrete_map={"Positive":"#22c55e","Neutral":"#f97316","Negative":"#ef4444"},
        )
        bd.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
            height=300, margin=dict(t=50,b=10,l=10,r=10), title_font_size=14,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        st.plotly_chart(bd, use_container_width=True)

    with ch2:
        # Sentiment by category within brand
        cat_sent = (
            brand_rev_t2.groupby("category")
            .agg(avg_compound=("compound","mean"), total=("compound","count"))
            .reset_index()
            .sort_values("avg_compound")
        )
        bar_cat = px.bar(
            cat_sent, x="avg_compound", y="category",
            orientation="h",
            title=f"{sel_brand} — Sentiment by Category",
            color="avg_compound",
            color_continuous_scale=["#ef4444","#f97316","#22c55e"],
            labels={"avg_compound":"Avg VADER","category":""},
        )
        bar_cat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", height=300,
            margin=dict(t=50,b=10,l=10,r=10),
            title_font_size=14, coloraxis_showscale=False,
        )
        st.plotly_chart(bar_cat, use_container_width=True)

    # Top products by sentiment
    prod_sent = (
        brand_rev_t2.groupby("title")
        .agg(avg_compound=("compound","mean"), total=("compound","count"))
        .reset_index()
        .sort_values("avg_compound", ascending=True)
        .tail(10)
    )
    prod_sent["short_title"] = prod_sent["title"].str[:50]
    bar2 = px.bar(
        prod_sent, x="avg_compound", y="short_title",
        orientation="h",
        title="Top Products by Avg Sentiment Score",
        color="avg_compound",
        color_continuous_scale=["#ef4444","#f97316","#22c55e"],
        labels={"avg_compound":"Avg VADER Score","short_title":""},
    )
    bar2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", height=340,
        margin=dict(t=50,b=10,l=10,r=10),
        title_font_size=14, coloraxis_showscale=False,
    )
    st.plotly_chart(bar2, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    col_pos, col_neg = st.columns(2)

    with col_pos:
        pos_brand = brand_rev_t2[brand_rev_t2["sentiment"] == "Positive"].sort_values("compound", ascending=False)
        st.markdown(
            f"<div class='section-pos'>✅ Positive Reviews — {len(pos_brand)}</div>",
            unsafe_allow_html=True,
        )
        max_bp = min(len(pos_brand), 30)
        show_bp = st.slider("Show top N positive", 1, max_bp, min(len(pos_brand), 8), key="bp_slider") if max_bp > 1 else max_bp
        for _, row in pos_brand.head(show_bp).iterrows():
            sv  = row["star_rating"]
            ss  = (str(round(sv,1)) + "/5") if pd.notna(sv) else "N/A"
            cv  = round(row["compound"], 3)
            ttl = str(row["title"])[:55]
            cat = str(row.get("category",""))
            st.markdown(f"""
<div class="pos-card">
  <div style="color:#4ade80;font-size:11px;margin-bottom:4px;">{CATEGORY_ICON.get(cat,'⚙️')} {ttl}</div>
  <div class="review-text">{row['review_text']}</div>
  <div class="review-meta">
    <span class="star-pos">★ {ss}</span> &nbsp;
    VADER: <span class="score-pos">{cv:+.3f}</span>
  </div>
</div>""", unsafe_allow_html=True)

    with col_neg:
        neg_brand = brand_rev_t2[brand_rev_t2["sentiment"] == "Negative"].sort_values("compound", ascending=True)
        st.markdown(
            f"<div class='section-neg'>❌ Negative Reviews — {len(neg_brand)}</div>",
            unsafe_allow_html=True,
        )
        max_bn = min(len(neg_brand), 30)
        show_bn = st.slider("Show top N negative", 1, max_bn, min(len(neg_brand), 8), key="bn_slider") if max_bn > 1 else max_bn
        for _, row in neg_brand.head(show_bn).iterrows():
            sv  = row["star_rating"]
            ss  = (str(round(sv,1)) + "/5") if pd.notna(sv) else "N/A"
            cv  = round(row["compound"], 3)
            ttl = str(row["title"])[:55]
            cat = str(row.get("category",""))
            st.markdown(f"""
<div class="neg-card">
  <div style="color:#f87171;font-size:11px;margin-bottom:4px;">{CATEGORY_ICON.get(cat,'⚙️')} {ttl}</div>
  <div class="review-text">{row['review_text']}</div>
  <div class="review-meta">
    <span class="star-neg">★ {ss}</span> &nbsp;
    VADER: <span class="score-neg">{cv:+.3f}</span>
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — BROWSE BY CATEGORY  (NEW)
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 📂 Browse Electronics by Category")

    avail_cats_t3 = sorted(fdf["category"].dropna().unique().tolist())
    sel_cat_t3 = st.selectbox(
        "Select a product category",
        avail_cats_t3,
        key="cat_tab3",
        format_func=lambda c: f"{CATEGORY_ICON.get(c,'⚙️')}  {c}",
    )

    cat_df  = fdf[fdf["category"] == sel_cat_t3]
    cat_rev = frev[frev["category"] == sel_cat_t3]

    c_total = len(cat_rev)
    c_pos   = (cat_rev["sentiment"] == "Positive").sum()
    c_neg   = (cat_rev["sentiment"] == "Negative").sum()
    c_neu   = (cat_rev["sentiment"] == "Neutral").sum()
    c_safe  = c_total if c_total > 0 else 1

    cicon = CATEGORY_ICON.get(sel_cat_t3, "⚙️")
    st.markdown(f"## {cicon} {sel_cat_t3}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Products</div>
  <div class="kpi-value" style="color:#60a5fa;">{len(cat_df)}</div>
</div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Brands</div>
  <div class="kpi-value" style="color:#60a5fa;">{cat_df['manufacturer'].nunique()}</div>
</div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Positive</div>
  <div class="kpi-value" style="color:#22c55e;">{c_pos:,}</div>
  <div class="kpi-sub">{c_pos/c_safe*100:.1f}%</div>
</div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Negative</div>
  <div class="kpi-value" style="color:#ef4444;">{c_neg:,}</div>
  <div class="kpi-sub">{c_neg/c_safe*100:.1f}%</div>
</div>""", unsafe_allow_html=True)
    c5.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Avg VADER</div>
  <div class="kpi-value" style="color:#a78bfa;">{cat_rev['compound'].mean() if c_total > 0 else 0.0:+.3f}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    cc1, cc2 = st.columns(2)

    with cc1:
        cat_donut = px.pie(
            values=[c_pos, c_neu, c_neg],
            names=["Positive","Neutral","Negative"],
            hole=0.55,
            title=f"{sel_cat_t3} — Sentiment Split",
            color_discrete_map={"Positive":"#22c55e","Neutral":"#f97316","Negative":"#ef4444"},
        )
        cat_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
            height=300, margin=dict(t=50,b=10,l=10,r=10), title_font_size=14,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        st.plotly_chart(cat_donut, use_container_width=True)

    with cc2:
        # Brand comparison within category
        brand_in_cat = (
            cat_rev.groupby("manufacturer")
            .agg(avg_compound=("compound","mean"), total=("compound","count"))
            .reset_index()
            .sort_values("avg_compound")
        )
        brand_bar = px.bar(
            brand_in_cat, x="avg_compound", y="manufacturer",
            orientation="h",
            title=f"Brand Sentiment in {sel_cat_t3}",
            color="avg_compound",
            color_continuous_scale=["#ef4444","#f97316","#22c55e"],
            labels={"avg_compound":"Avg VADER","manufacturer":""},
        )
        brand_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", height=300,
            margin=dict(t=50,b=10,l=10,r=10),
            title_font_size=14, coloraxis_showscale=False,
        )
        st.plotly_chart(brand_bar, use_container_width=True)

    # Price vs rating scatter
    if cat_df["price_clean"].notna().sum() > 3:
        fig_scatter = px.scatter(
            cat_df.dropna(subset=["price_clean","rating_clean"]),
            x="price_clean", y="rating_clean",
            color="manufacturer",
            hover_name="title",
            title=f"Price vs Rating — {sel_cat_t3}",
            labels={"price_clean":"Price (₹)","rating_clean":"Rating (/ 5)"},
            opacity=0.75,
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", height=340, title_font_size=14,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Top / Bottom products
    top_col, bot_col = st.columns(2)

    with top_col:
        st.markdown("<div class='section-pos'>🏆 Top Rated Products</div>", unsafe_allow_html=True)
        top_prods = (
            cat_df.dropna(subset=["rating_clean"])
            .sort_values("rating_clean", ascending=False)
            [["title","manufacturer","price_clean","rating_clean"]]
            .head(10)
            .copy()
        )
        top_prods.columns = ["Product","Brand","Price (₹)","Rating"]
        top_prods["Price (₹)"] = top_prods["Price (₹)"].apply(lambda x: f"₹{int(x):,}" if pd.notna(x) else "N/A")
        top_prods["Rating"] = top_prods["Rating"].apply(lambda x: f"{x:.1f} / 5")
        st.dataframe(top_prods.reset_index(drop=True), use_container_width=True, hide_index=True)

    with bot_col:
        st.markdown("<div class='section-neg'>💬 Best Sentiment (by VADER)</div>", unsafe_allow_html=True)
        sent_prods = (
            cat_rev.groupby("title")
            .agg(avg_vader=("compound","mean"), reviews=("compound","count"))
            .reset_index()
            .sort_values("avg_vader", ascending=False)
            .head(10)
        )
        sent_prods["avg_vader"] = sent_prods["avg_vader"].round(3)
        sent_prods.columns = ["Product","Avg VADER","Review Count"]
        sent_prods["Product"] = sent_prods["Product"].str[:50]
        st.dataframe(sent_prods.reset_index(drop=True), use_container_width=True, hide_index=True)

    # Positive / Negative reviews for category
    st.markdown("<hr>", unsafe_allow_html=True)
    cr1, cr2 = st.columns(2)
    with cr1:
        pos_cat_r = cat_rev[cat_rev["sentiment"] == "Positive"].sort_values("compound", ascending=False)
        st.markdown(f"<div class='section-pos'>✅ Positive Reviews — {len(pos_cat_r)}</div>", unsafe_allow_html=True)
        mc = min(len(pos_cat_r), 30)
        sc = st.slider("Positive reviews to show", 1, mc, min(len(pos_cat_r),6), key="pos_cat") if mc > 1 else mc
        for _, row in pos_cat_r.head(sc).iterrows():
            sv  = row["star_rating"]
            ss  = (str(round(sv,1)) + "/5") if pd.notna(sv) else "N/A"
            cv  = round(row["compound"], 3)
            ttl = str(row["title"])[:55]
            st.markdown(f"""
<div class="pos-card">
  <div style="color:#4ade80;font-size:11px;margin-bottom:4px;">{row['manufacturer']} — {ttl}</div>
  <div class="review-text">{row['review_text']}</div>
  <div class="review-meta">
    <span class="star-pos">★ {ss}</span> &nbsp;
    VADER: <span class="score-pos">{cv:+.3f}</span>
  </div>
</div>""", unsafe_allow_html=True)

    with cr2:
        neg_cat_r = cat_rev[cat_rev["sentiment"] == "Negative"].sort_values("compound", ascending=True)
        st.markdown(f"<div class='section-neg'>❌ Negative Reviews — {len(neg_cat_r)}</div>", unsafe_allow_html=True)
        mn = min(len(neg_cat_r), 30)
        sn = st.slider("Negative reviews to show", 1, mn, min(len(neg_cat_r),6), key="neg_cat") if mn > 1 else mn
        for _, row in neg_cat_r.head(sn).iterrows():
            sv  = row["star_rating"]
            ss  = (str(round(sv,1)) + "/5") if pd.notna(sv) else "N/A"
            cv  = round(row["compound"], 3)
            ttl = str(row["title"])[:55]
            st.markdown(f"""
<div class="neg-card">
  <div style="color:#f87171;font-size:11px;margin-bottom:4px;">{row['manufacturer']} — {ttl}</div>
  <div class="review-text">{row['review_text']}</div>
  <div class="review-meta">
    <span class="star-neg">★ {ss}</span> &nbsp;
    VADER: <span class="score-neg">{cv:+.3f}</span>
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4 — OVERALL ANALYTICS
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### 📊 Overall Sentiment Analytics — All Electronics")

    # ── Category-level overview ──────────────
    st.markdown("#### Sentiment by Product Category")
    cat_stats = (
        frev.groupby("category")
        .agg(
            pos=("sentiment", lambda x: (x=="Positive").sum()),
            neg=("sentiment", lambda x: (x=="Negative").sum()),
            total=("sentiment","count"),
            avg_score=("compound","mean"),
        )
        .reset_index()
        .sort_values("avg_score", ascending=False)
    )
    cat_stats["pos_pct"] = (cat_stats["pos"] / cat_stats["total"] * 100).round(1)
    cat_stats["neg_pct"] = (cat_stats["neg"] / cat_stats["total"] * 100).round(1)
    cat_stats["icon"]    = cat_stats["category"].map(lambda c: CATEGORY_ICON.get(c,"⚙️"))
    cat_stats["cat_label"] = cat_stats["icon"] + "  " + cat_stats["category"]

    fig_cat_bar = px.bar(
        cat_stats, x="cat_label", y=["pos_pct","neg_pct"],
        barmode="group",
        title="Positive vs Negative % by Product Category",
        labels={"value":"Percentage (%)","cat_label":"Category","variable":"Sentiment"},
        color_discrete_map={"pos_pct":"#22c55e","neg_pct":"#ef4444"},
    )
    fig_cat_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", height=420,
        xaxis_tickangle=-35, title_font_size=15, legend_title_text="",
    )
    newnames = {"pos_pct":"Positive %","neg_pct":"Negative %"}
    fig_cat_bar.for_each_trace(lambda t: t.update(name=newnames[t.name]))
    st.plotly_chart(fig_cat_bar, use_container_width=True)

    # Avg VADER by category
    fig_cat_avg = px.bar(
        cat_stats.sort_values("avg_score"),
        x="avg_score", y="cat_label",
        orientation="h",
        title="Average VADER Score by Product Category",
        color="avg_score",
        color_continuous_scale=["#ef4444","#f97316","#22c55e"],
        labels={"avg_score":"Avg VADER","cat_label":""},
    )
    fig_cat_avg.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", height=520,
        title_font_size=15, coloraxis_showscale=False,
        margin=dict(l=130),
    )
    st.plotly_chart(fig_cat_avg, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Brand-level overview ─────────────────
    st.markdown("#### Sentiment by Brand")
    brand_stats = (
        frev.groupby("manufacturer")
        .agg(
            pos=("sentiment", lambda x: (x=="Positive").sum()),
            neg=("sentiment", lambda x: (x=="Negative").sum()),
            total=("sentiment","count"),
            avg_score=("compound","mean"),
        )
        .reset_index()
        .sort_values("avg_score", ascending=False)
    )
    brand_stats["pos_pct"] = (brand_stats["pos"] / brand_stats["total"] * 100).round(1)
    brand_stats["neg_pct"] = (brand_stats["neg"] / brand_stats["total"] * 100).round(1)

    fig_brand = px.bar(
        brand_stats.sort_values("avg_score", ascending=False).head(40),
        x="manufacturer", y=["pos_pct","neg_pct"],
        barmode="group",
        title="Positive vs Negative Review % by Brand (Top 40)",
        labels={"value":"Percentage (%)","manufacturer":"Brand","variable":"Sentiment"},
        color_discrete_map={"pos_pct":"#22c55e","neg_pct":"#ef4444"},
    )
    fig_brand.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", height=420,
        xaxis_tickangle=-35, title_font_size=15, legend_title_text="",
    )
    fig_brand.for_each_trace(lambda t: t.update(name=newnames[t.name]))
    st.plotly_chart(fig_brand, use_container_width=True)

    # Avg VADER heatmap (brand × category)
    st.markdown("#### Brand × Category Sentiment Heatmap")
    pivot_data = (
        frev.groupby(["manufacturer","category"])["compound"]
        .mean()
        .reset_index()
        .pivot(index="manufacturer", columns="category", values="compound")
        .fillna(0)
    )
    # Limit to top 30 brands by total reviews for readability
    top_brands_hm = (
        frev.groupby("manufacturer")["compound"].count()
        .sort_values(ascending=False)
        .head(30).index.tolist()
    )
    pivot_data = pivot_data.loc[pivot_data.index.isin(top_brands_hm)]
    pivot_data = pivot_data[[c for c in pivot_data.columns if c in CATEGORIES]]

    fig_hm = px.imshow(
        pivot_data,
        color_continuous_scale=["#ef4444","#1e1e2e","#22c55e"],
        zmin=-0.5, zmax=0.5,
        title="Avg VADER Sentiment — Brand × Category (Top 30 Brands)",
        labels={"color":"Avg VADER"},
        aspect="auto",
    )
    fig_hm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", height=700, title_font_size=15,
        xaxis_tickangle=-40, margin=dict(l=100),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # VADER Distribution
    fig_dist = px.histogram(
        frev, x="compound", nbins=80,
        color="sentiment",
        title="VADER Score Distribution — All Reviews",
        color_discrete_map={"Positive":"#22c55e","Neutral":"#f97316","Negative":"#ef4444"},
        labels={"compound":"VADER Compound Score"},
        barmode="overlay", opacity=0.75,
    )
    fig_dist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", height=340, title_font_size=15,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Summary tables
    st.markdown("#### Category Sentiment Summary")
    disp_cat = cat_stats[["category","total","pos","neg","pos_pct","neg_pct","avg_score"]].copy()
    disp_cat.columns = ["Category","Total Reviews","Positive","Negative","Positive %","Negative %","Avg VADER"]
    disp_cat["Avg VADER"] = disp_cat["Avg VADER"].round(3)
    st.dataframe(disp_cat.sort_values("Positive %", ascending=False).reset_index(drop=True),
                 use_container_width=True, hide_index=True)

    st.markdown("#### Brand Sentiment Summary (All Brands)")
    disp_brand = brand_stats[["manufacturer","total","pos","neg","pos_pct","neg_pct","avg_score"]].copy()
    disp_brand.columns = ["Brand","Total Reviews","Positive","Negative","Positive %","Negative %","Avg VADER"]
    disp_brand["Avg VADER"] = disp_brand["Avg VADER"].round(3)
    st.dataframe(disp_brand.sort_values("Positive %", ascending=False).reset_index(drop=True),
                 use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center; color:#475569; font-size:12px; padding: 8px 0;">
  🔌 Electronics Sentiment Analyser &nbsp;|&nbsp; VADER NLP Engine &nbsp;|&nbsp;
  Enhanced_Dataset_Final.csv &nbsp;|&nbsp;
  {len(df):,} Products &nbsp;·&nbsp; {df['manufacturer'].nunique()} Brands &nbsp;·&nbsp;
  {df['category'].nunique()} Categories &nbsp;·&nbsp; {TOTAL_REVIEWS:,} Reviews Analysed
</div>
""", unsafe_allow_html=True)