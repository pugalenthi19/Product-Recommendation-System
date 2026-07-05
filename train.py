"""
=======================================================================
COMPLETE ML PIPELINE — OPTIMIZED
5 Models × All Columns (including Reviews NLP)  ✚  VISUALIZATIONS
=======================================================================
  ► Run in Google Colab
  ► First cell (once):  !pip install squarify -q
=======================================================================
"""

# ── COLAB INSTALL (uncomment & run once) ──────────────────────────
# !pip install squarify -q

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, mean_squared_error, r2_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                               RandomForestClassifier)
from sklearn.impute import SimpleImputer
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────
df = pd.read_csv('Enhanced_Dataset_Final_1.csv')
df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)

# ─────────────────────────────────────────────────────────────
# PARSERS
# ─────────────────────────────────────────────────────────────
def parse_num(val):
    if pd.isna(val): return np.nan
    m = re.search(r'(\d+\.?\d*)', str(val))
    return float(m.group(1)) if m else np.nan

def parse_rev_rating(val):
    if pd.isna(val): return np.nan
    m = re.findall(r'(\d+\.?\d*)\s*out of 5', str(val))
    return np.mean([float(x) for x in m]) if m else np.nan

# ─────────────────────────────────────────────────────────────
# FAST RECOMMENDATION (vectorized cosine similarity)
# ─────────────────────────────────────────────────────────────
def content_based(X, y, task, n=300, k=5, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    Xs, ys = X[idx], y[idx]
    sim = cosine_similarity(Xs)
    np.fill_diagonal(sim, -1)
    top_k = np.argsort(sim, axis=1)[:, -k:]
    if task == 'classification':
        preds = np.array([np.bincount(ys[top_k[i]].astype(int)).argmax() for i in range(len(Xs))])
        return float(np.mean(preds == ys))
    else:
        preds = ys[top_k].mean(axis=1)
        denom = np.abs(ys) + 1e-6
        return float(np.mean(np.maximum(0, 1 - np.abs(preds - ys) / denom)))

def collab_filter(X, y, task, n=600, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    Xs, ys = X[idx], y[idx]
    nc = min(15, Xs.shape[1]-1)
    svd = TruncatedSVD(n_components=nc, random_state=42)
    Xl = svd.fit_transform(Xs)
    Xtr, Xte, ytr, yte = train_test_split(Xl, ys, test_size=0.2, random_state=42)
    if task == 'classification':
        m = KNeighborsClassifier(n_neighbors=7)
        m.fit(Xtr, ytr.astype(int))
        return float(m.score(Xte, yte.astype(int)))
    else:
        m = KNeighborsRegressor(n_neighbors=7)
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        return float(max(0, r2_score(yte, p)))

# ─────────────────────────────────────────────────────────────
# FEATURE BUILDER
# ─────────────────────────────────────────────────────────────
def build_X(df, excl):
    d = df.copy()
    d['_price']  = d['price'].apply(parse_num)
    d['_rating'] = d['rating'].apply(parse_num)
    d['_revn']   = d['total_reviews'].apply(parse_num)
    d['_weight'] = d['weight'].apply(parse_num)
    d['_batt']   = pd.to_numeric(d['battery_power_rating'], errors='coerce')
    d['_ravg']   = d['reviews_rating'].apply(parse_rev_rating)
    d['_rcnt']   = d['reviews'].apply(lambda x: len(str(x).split('||')) if pd.notna(x) else 0)

    num_f = [c for c in ['_price','_rating','_revn','_weight','_batt','_ravg','_rcnt'] if c != excl]
    X_n = SimpleImputer(strategy='median').fit_transform(d[num_f].values.astype(float))

    cat_f = [c for c in ['availability_status','manufacturer','country_of_origin',
                          'os','form_factor','colour','ram'] if c != excl]
    X_c = np.hstack([LabelEncoder().fit_transform(d[c].fillna('Unknown').astype(str)).reshape(-1,1)
                     for c in cat_f]) if cat_f else np.zeros((len(d),1))

    X_t = (TfidfVectorizer(max_features=60, stop_words='english')
           .fit_transform(d['title'].fillna('').astype(str)).toarray()
           if excl != 'title' else np.zeros((len(d),5)))

    return StandardScaler().fit_transform(np.hstack([X_n, X_c, X_t]))

# ─────────────────────────────────────────────────────────────
# TARGET PREP
# ─────────────────────────────────────────────────────────────
def prep_y(df, col):
    d = df[col].copy()
    if col in ('price','total_reviews','weight','battery_power_rating'):
        y = d.apply(parse_num); mask = y.notna()
        return y[mask].values, 'regression', mask.values

    if col in ('rating','reviews_rating'):
        fn = parse_num if col == 'rating' else parse_rev_rating
        y = d.apply(fn); mask = y.notna()
        yc = pd.cut(y[mask], bins=[0,2,3,4,5.01], labels=[0,1,2,3], include_lowest=True)
        return yc.astype(float).fillna(1).astype(int).values, 'classification', mask.values

    if col == 'reviews':
        y = d.apply(lambda x: len(str(x).split('||')) if pd.notna(x) else 0)
        return y.values, 'regression', np.ones(len(d), bool)

    ys = d.fillna('Unknown').astype(str)
    top = ys.value_counts().nlargest(15).index.tolist()
    ys = ys.where(ys.isin(top), 'Other')
    cnt = ys.value_counts(); valid = cnt[cnt >= 4].index
    mask = ys.isin(valid)
    return LabelEncoder().fit_transform(ys[mask]), 'classification', mask.values

# ─────────────────────────────────────────────────────────────
# TRAIN ALL 5 MODELS
# ─────────────────────────────────────────────────────────────
def train_col(df, col):
    try:
        y, task, vmask = prep_y(df, col)
        X = build_X(df, col)[vmask]
        if len(np.unique(y)) < 2 or len(y) < 40: return None, None

        strat = None
        if task == 'classification':
            cnt = np.bincount(y.astype(int))
            if np.min(cnt) >= 2: strat = y

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
        R = []

        if task == 'regression':
            m = Ridge(alpha=0.5); m.fit(Xtr, ytr); p = m.predict(Xte)
            sc = max(0, r2_score(yte, p))
            R.append(('Linear Regression (Ridge)', sc, None, None, None, np.sqrt(mean_squared_error(yte, p))))
        else:
            m = LogisticRegression(C=10, max_iter=500, random_state=42, solver='lbfgs')
            m.fit(Xtr, ytr); p = m.predict(Xte)
            sc = accuracy_score(yte, p)
            R.append(('Linear (Logistic) Regression', sc,
                      precision_score(yte,p,average='weighted',zero_division=0),
                      recall_score(yte,p,average='weighted',zero_division=0),
                      f1_score(yte,p,average='weighted',zero_division=0),
                      np.sqrt(mean_squared_error(yte, p))))

        cb = content_based(X, y, task)
        R.append(('Content-Based Recommendation', cb, cb, cb, cb, None))

        cf = collab_filter(X, y, task)
        R.append(('Collaborative Filtering', cf, cf, cf, cf, None))

        hyb = 0.5*cb + 0.5*cf
        R.append(('Hybrid Recommendation', hyb, hyb, hyb, hyb, None))

        if task == 'regression':
            m2 = GradientBoostingRegressor(n_estimators=80, max_depth=4, random_state=42)
            m2.fit(Xtr, ytr); p2 = m2.predict(Xte)
            sc2 = max(0, r2_score(yte, p2))
            R.append(('GBM Regressor', sc2, None, None, None, np.sqrt(mean_squared_error(yte, p2))))
        else:
            m2 = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
            m2.fit(Xtr, ytr); p2 = m2.predict(Xte)
            sc2 = accuracy_score(yte, p2)
            R.append(('Random Forest', sc2,
                      precision_score(yte,p2,average='weighted',zero_division=0),
                      recall_score(yte,p2,average='weighted',zero_division=0),
                      f1_score(yte,p2,average='weighted',zero_division=0),
                      np.sqrt(mean_squared_error(yte, p2))))
        return R, task
    except Exception as e:
        print(f"  [SKIP] {col}: {e}")
        return None, None

# ─────────────────────────────────────────────────────────────
# REVIEWS NLP
# ─────────────────────────────────────────────────────────────
def run_reviews(df):
    print("\n" + "="*100)
    print("REVIEWS COLUMN — NLP SENTIMENT PIPELINE")
    print("="*100)

    d = df.copy()
    d['ravg'] = d['reviews_rating'].apply(parse_rev_rating).fillna(
                 d['reviews_rating'].apply(parse_rev_rating).median())
    d['sent'] = pd.cut(d['ravg'], bins=[0,2.5,3.5,5.01], labels=[0,1,2],
                       include_lowest=True).astype(float).fillna(1).astype(int)

    rev_text = d['reviews'].fillna('no review').astype(str)
    Xrev    = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1,2)).fit_transform(rev_text)
    Xtitle  = TfidfVectorizer(max_features=80,  stop_words='english').fit_transform(d['title'].fillna('').astype(str))
    Xnum    = csr_matrix(np.column_stack([
                  d['price'].apply(parse_num).fillna(0),
                  d['rating'].apply(parse_num).fillna(0),
                  rev_text.apply(lambda x: len(x.split('||')))
              ]))
    X = hstack([Xrev, Xtitle, Xnum])
    y = d['sent'].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    R = []

    lr = LogisticRegression(C=5, max_iter=500, random_state=42, solver='lbfgs')
    lr.fit(Xtr, ytr); p = lr.predict(Xte)
    sc = accuracy_score(yte, p)
    R.append(('Logistic Regression (Linear)', sc,
              precision_score(yte,p,average='weighted',zero_division=0),
              recall_score(yte,p,average='weighted',zero_division=0),
              f1_score(yte,p,average='weighted',zero_division=0),
              np.sqrt(mean_squared_error(yte, p))))

    Xd = X.toarray()[:1500]; yd = y[:1500]
    cb  = content_based(Xd, yd, 'classification')
    cf  = collab_filter(Xd, yd, 'classification')
    hyb = 0.5*cb + 0.5*cf
    R.append(('Content-Based Recommendation', cb,  cb,  cb,  cb,  None))
    R.append(('Collaborative Filtering',      cf,  cf,  cf,  cf,  None))
    R.append(('Hybrid Recommendation',        hyb, hyb, hyb, hyb, None))

    svm = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    svm.fit(Xtr, ytr); p2 = svm.predict(Xte)
    sc2 = accuracy_score(yte, p2)
    R.append(('SVM (LinearSVC)', sc2,
              precision_score(yte,p2,average='weighted',zero_division=0),
              recall_score(yte,p2,average='weighted',zero_division=0),
              f1_score(yte,p2,average='weighted',zero_division=0),
              np.sqrt(mean_squared_error(yte, p2))))

    print_tbl(R, 'classification', 'reviews')
    return R

# ─────────────────────────────────────────────────────────────
# PRINT
# ─────────────────────────────────────────────────────────────
W = 100
def print_tbl(R, task, col):
    best = max(r[1] for r in R)
    best_nm = next(r[0] for r in R if r[1] == best)
    print(f"\nTarget: {col.upper()}   Task: {task.upper()}")
    if task == 'classification':
        print(f"{'Model':<38} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>9} {'RMSE':>9}")
        print("-"*W)
        for nm,sc,pr,re_,f1,rm in R:
            star = " ★" if nm == best_nm else "  "
            print(f"{nm+star:<38} {sc*100:>8.2f}%"
                  f" {(str(round(pr*100,2))+'%') if pr else 'N/A':>10}"
                  f" {(str(round(re_*100,2))+'%') if re_ else 'N/A':>8}"
                  f" {(str(round(f1*100,2))+'%') if f1 else 'N/A':>9}"
                  f" {(str(round(rm,4))) if rm else 'N/A':>9}")
    else:
        print(f"{'Model':<38} {'R²/Score':>10} {'RMSE':>15}")
        print("-"*65)
        for nm,sc,pr,re_,f1,rm in R:
            star = " ★" if nm == best_nm else "  "
            print(f"{nm+star:<38} {sc*100:>9.2f}%  {str(round(rm,4)) if rm else 'N/A':>15}")
    print(f"\n  ★ Best: {best_nm}  |  Score: {best*100:.2f}%")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
results_store = {}          # ← stores all results for visualization

all_cols = [c for c in df.columns if c != 'reviews']
summary  = []

print(f"\nDataset: {df.shape[0]} rows × {df.shape[1]} cols")
print("Columns:", df.columns.tolist())
print("\n" + "="*100)
print("ML PIPELINE — ALL TARGET COLUMNS")
print("="*100)

for i, col in enumerate(all_cols, 1):
    print(f"\n{'='*100}")
    print(f"RUN {i}/{len(all_cols)}  ──  TARGET: {col}")
    print(f"{'='*100}")
    R, task = train_col(df, col)
    if R is None:
        print("  [SKIPPED — insufficient variation]")
        continue
    print_tbl(R, task, col)
    results_store[col] = (R, task)              # ← store for visualization
    best_sc = max(r[1] for r in R)
    best_nm = next(r[0] for r in R if r[1] == best_sc)
    summary.append((col, task, best_nm, best_sc))

# Reviews NLP
R_rev = run_reviews(df)
results_store['reviews'] = (R_rev, 'classification (NLP)')  # ← store for visualization
best_rev = max(r[1] for r in R_rev)
best_rev_nm = next(r[0] for r in R_rev if r[1] == best_rev)
summary.append(('reviews', 'classification (NLP)', best_rev_nm, best_rev))

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY TABLE (console)
# ─────────────────────────────────────────────────────────────
print("\n\n" + "="*105)
print("FINAL SUMMARY  —  BEST MODEL PER TARGET COLUMN")
print("="*105)
print(f"{'#':<4} {'Target Column':<28} {'Task':<24} {'Best Model':<38} {'Score':>8}")
print("-"*105)
for j,(col,task,nm,sc) in enumerate(summary, 1):
    print(f"{j:<4} {col:<28} {task:<24} {nm:<38} {sc*100:>7.2f}%")
avg = np.mean([s[3] for s in summary])
print("-"*105)
print(f"\n  AVERAGE ACCURACY (best model per column): {avg*100:.2f}%")
print(f"  Columns evaluated: {len(summary)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  V I S U A L I Z A T I O N   S E C T I O N
#  18 different chart types — one per column + final comparative chart
# ═══════════════════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy.interpolate import make_interp_spline

try:
    import squarify
except Exception:
    import subprocess
    subprocess.run(['pip', 'install', 'squarify', '-q'], capture_output=True)
    import squarify

# ── Global dark theme ─────────────────────────────────────────────────────────
BG     = '#0D1117'
PANEL  = '#161B22'
GRID_C = '#21262D'
TEXT   = '#E6EDF3'
SUB    = '#8B949E'

PALETTE = [
    '#58A6FF', '#3FB950', '#F78166', '#D2A8FF', '#FFA657',
    '#79C0FF', '#56D364', '#FF7B72', '#BC8CFF', '#E3B341'
]

MODEL_SHORT = {
    'Linear Regression (Ridge)':    'Ridge',
    'Linear (Logistic) Regression': 'Logistic',
    'Logistic Regression (Linear)': 'Logistic',
    'Content-Based Recommendation': 'Content-Based',
    'Collaborative Filtering':      'Collab. Filter',
    'Hybrid Recommendation':        'Hybrid',
    'SVM / GBM Regressor':          'GBM',
    'SVM / Random Forest':          'Rnd Forest',
    'SVM (LinearSVC)':              'LinearSVC',
}

MODEL_COLORS_MAP = {
    'Ridge':          '#58A6FF',
    'Logistic':       '#58A6FF',
    'Content-Based':  '#3FB950',
    'Collab. Filter': '#F78166',
    'Hybrid':         '#D2A8FF',
    'GBM':            '#FFA657',
    'Rnd Forest':     '#FFA657',
    'LinearSVC':      '#FFA657',
}

plt.rcParams.update({
    'figure.facecolor':   BG,
    'axes.facecolor':     PANEL,
    'axes.edgecolor':     GRID_C,
    'axes.labelcolor':    TEXT,
    'xtick.color':        SUB,
    'ytick.color':        SUB,
    'text.color':         TEXT,
    'grid.color':         GRID_C,
    'grid.alpha':         0.5,
    'font.family':        'monospace',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

def get_data(col):
    R, task = results_store[col]
    names  = [MODEL_SHORT.get(r[0], r[0]) for r in R]
    scores = [r[1] * 100 for r in R]
    colors = [MODEL_COLORS_MAP.get(n, PALETTE[i % len(PALETTE)])
              for i, n in enumerate(names)]
    best_i = int(np.argmax(scores))
    return names, scores, colors, best_i, task


# ─────────────────────────────────────────────────────────────────────────────
# CHART TYPE 01 — HORIZONTAL BAR
# ─────────────────────────────────────────────────────────────────────────────
def chart_horizontal_bar(ax, names, scores, colors, col, best_i):
    y = np.arange(len(names))
    bars = ax.barh(y, scores, color=colors, height=0.55, linewidth=0, zorder=2)
    for i, (bar, sc) in enumerate(zip(bars, scores)):
        ax.text(sc + 0.8, i, f'{sc:.1f}%', va='center', fontsize=9.5,
                color=TEXT, fontweight='bold' if i == best_i else 'normal')
        if i == best_i:
            bar.set_edgecolor('#FFD700'); bar.set_linewidth(1.8)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlabel('Score (%)', fontsize=9); ax.set_xlim(0, 118)
    ax.set_title(f'📊  OUTPUT FEATURE : {col.upper()}', color=TEXT, fontsize=12, pad=10, fontweight='bold')
    ax.grid(axis='x', zorder=0); ax.set_axisbelow(True)
    _best_tag(ax, names[best_i], scores[best_i])

# ─────────────────────────────────────────────────────────────────────────────
# CHART TYPE 02 — PIE CHART
# ─────────────────────────────────────────────────────────────────────────────
def chart_pie(ax, names, scores, colors, col, best_i):
    explode = [0.07 if i == best_i else 0 for i in range(len(scores))]
    wedges, texts, autotexts = ax.pie(
        scores, labels=names, colors=colors, autopct='%1.1f%%',
        explode=explode, startangle=140, pctdistance=0.78,
        wedgeprops=dict(edgecolor=BG, linewidth=1.8))
    for t in texts:    t.set_fontsize(8.5); t.set_color(TEXT)
    for at in autotexts: at.set_fontsize(8); at.set_color(BG); at.set_fontweight('bold')
    ax.set_title(f'🥧  OUTPUT FEATURE : {col.upper()}', color=TEXT, fontsize=12, pad=10, fontweight='bold')
    _best_tag(ax, names[best_i], scores[best_i], pie=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHART TYPE 03 — RADAR / SPIDER
# ─────────────────────────────────────────────────────────────────────────────
def chart_radar(ax, names, scores, colors, col, best_i):
    N = len(names)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    sc_r   = scores + [scores[0]]
    ang_r  = angles  + [angles[0]]
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.plot(ang_r, sc_r, 'o-', linewidth=2.2, color=PALETTE[3], markersize=7)
    ax.fill(ang_r, sc_r, alpha=0.22, color=PALETTE[3])
    ax.set_xticks(angles); ax.set_xticklabels(names, fontsize=9, color=TEXT)
    ax.set_ylim(0, 115); ax.set_facecolor(PANEL)
    ax.tick_params(colors=SUB)
    for i, (a, sc) in enumerate(zip(angles, scores)):
        fw = 'bold' if i == best_i else 'normal'
        ax.text(a, sc + 7, f'{sc:.1f}%', ha='center', va='center', fontsize=8, color=TEXT, fontweight=fw)
    ax.set_title(f'🕸️  OUTPUT FEATURE : {col.upper()}', color=TEXT, fontsize=12, pad=22, fontweight='bold')

# ─────────────────────────────────────────────────────────────────────────────
# CHART TYPE 04 — LOLLIPOP
# ─────────────────────────────────────────────────────────────────────────────
def chart_lollipop(ax, names, scores, colors, col, best_i):
    y = np.arange(len(names))
    for i, (yi, sc, c) in enumerate(zip(y, scores, colors)):
        ax.plot([0, sc], [yi, yi], color=c, linewidth=2.8, alpha=0.75, zorder=1)
        ms = 200 if i == best_i else 110
        ec = '#FFD700' if i == best_i else 'none'
        ax.scatter([sc], [yi], color=c, s=ms, zorder=3, edgecolors=ec, linewidth=2)
        ax.text(sc + 1, yi, f'{sc:.1f}%', va='center', fontsize=9.5,
                color=TEXT, fontweight='bold' if i == best_i else 'normal')
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlabel('Score (%)', fontsize=9); ax.set_xlim(0, 118)
    ax.set_title(f'🍭  OUTPUT FEATURE : {col.upper()}', color=TEXT, fontsize=12, pad=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.3); _best_tag(ax, names[best_i], scores[best_i])

# ─────────────────────────────────────────────────────────────────────────────
# CHART TYPE 05 — DONUT
# ─────────────────────────────────────────────────────────────────────────────
def chart_donut(ax, names, scores, colors, col, best_i):
    wedges, _, autotexts = ax.pie(
        scores, colors=colors, autopct='%1.1f%%', startangle=90,
        pctdistance=0.82, wedgeprops=dict(width=0.45, edgecolor=BG, linewidth=2.2))
    for at in autotexts: at.set_fontsize(8); at.set_color(BG); at.set_fontweight('bold')
    ax.text(0,  0.10, '★ BEST',          ha='center', va='center', fontsize=9,  color='#FFD700', fontweight='bold')
    ax.text(0, -0.12, names[best_i],     ha='center', va='center', fontsize=8,  color=TEXT,      style='italic')
    ax.text(0, -0.32, f'{scores[best_i]:.1f}%', ha='center', va='center', fontsize=9.5, color=PALETTE[0], fontweight='bold')
    ax.set_title(f'🍩  OUTPUT FEATURE : {col.upper()}', color=TEXT, fontsize=12, pad=10, fontweight='bold')
    handles = [mpatches.Patch(color=colors[i], label=names[i]) for i in range(len(names))]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5,-0.22),
              ncol=3, fontsize=8, framealpha=0, labelcolor=TEXT)

# ─────────────────────────────────────────────────────────────────────────────
# CHART TYPE 06 — STEP LINE
# ─────────────────────────────────────────────────────────────────────────────
def chart_step_line(ax, names, scores, colors, col, best_i):
    x = np.arange(len(names))
    ax.step(x, scores, where='mid', color=PALETTE[0], linewidth=2.8, zorder=2)
    ax.fill_between(x, scores, step='mid', alpha=0.14, color=PALETTE[0])
    for i, (xi, sc) in enumerate(zip(x, scores)):
        ms = 120 if i == best_i else 60
        ec = '#FFD700' if i == best_i else 'none'
        ax.scatter([xi], [sc], color=colors[i], s=ms, zorder=4, edgecolors=ec, linewidth=2)
        ax.text(xi, sc + 2.5, f'{sc:.1f}%', ha='center', fontsize=9,
                color=TEXT, fontweight='bold' if i == best_i else 'normal')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5, rotation=20, ha='right')
    ax.set_ylabel('Score (%)', fontsize=9); ax.set_ylim(0, 118)
    ax.set_title(f'〰️  OUTPUT FEATURE : {col.upper()}', color=TEXT, fontsize=12, pad=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3); _best_tag(ax, names[best_i], scores[best_i])

# ─────────────────────────────────────────────────────────────────────────────
# CHART TYPE 07 — BUBBLE
# ─────────────────────────────────────────────────────────────────────────────
def chart_bubble(ax, names, scores, colors, col, best_i):
    x = np.arange(len(names))
    sizes = [max(400, s**1.85) for s in scores]
    lws   = [3 if i == best_i else 0.6 for i in range(len(names))]
    ecs   = ['#FFD700' if i == best_i else PANEL for i in range(len(names))]
    ax.scatter(x, scores, s=sizes, c=colors, alpha=0.85,
               edgecolors=ecs, linewidths=lws, zorder=3)
    for i, (xi, sc) in enumerate(zip(x, scores)):
        ax.text(xi, sc, f'{sc:.1f}%', ha='center', va='center',
                fontsize=8, color=BG, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5, rotation=22, ha='right')
    ax.set_ylabel('Score (%)', fontsize=9); ax.set_ylim(0, 120)
    ax.set_title(f'🫧  OUTPUT FEATURE : {col.upper()}', color=TEXT, fontsize=12, pad=10, fontweight='bold')
    ax.grid(alpha=0.3); _best_tag(ax, names[best_i], scores[best_i])

# ─────────────────────────────────────────────────────────────────────────────
# CHART TYPE 08 — FUNNEL
# ─────────────────────────────────────────────────────────────────────────────
def chart_funnel(ax, names, scores, colors, col, best_i):
    order  = np.argsort(scores)[::-1]
    snames = [names[i] for i in order]
    sscores= [scores[i] for i in order]
    scols  = [colors[i] for i in order]
    n = len(sscores)
    for i, (nm, sc, c) in enumerate(zip(snames, sscores, scols)):
        w    = sc / sscores[0]
        left = (1 - w) / 2
        top  = (n - 1 - i) / n + 0.05
        rect = FancyBboxPatch((left, top), w, 0.82/n,
                              boxstyle='round,pad=0.01', fc=c, ec=BG, linewidth=1.5,
                              zorder=2)
        ax.add_patch(rect)
        is_best = (order[i] == best_i)
        label = f'★ {nm}  {sc:.1f}%' if is_best else f'{nm}  {sc:.1f}%'
        ax.text(0.5, top + 0.41/n, label, ha='center', va='center',
                fontsize=9, color=BG, fontweight='bold')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.set_title(f'⬇️  OUTPUT FEATURE : {col.upper()}', color=TEXT, fontsize=12, pad=10, fontweight='bold')

# ─────────────────────────────────────────────────────────────────────────────
# CHART TYPE 09 — WAFFLE
# ─────────────────────────────────────────────────────────────────────────────
def chart_waffle(ax, names, scores, colors, col, best_i):
    total  = sum(scores)
    shares = [max(1, int(round(s / total * 100))) for s in scores]
    diff   = 100 - sum(shares)
    shares[int(np.argmax(scores))] += diff
    shares = np.clip(shares, 0, None)
    grid   = []
    for ci, sh in enumerate(shares):
        grid.extend([ci] * sh)
    grid = (grid + [0]*100)[:100]
    grid = np.array(grid).reshape(10, 10)
    for r in range(10):
        for c in range(10):
            v   = grid[r, c]
            clr = colors[v] if v < len(colors) else PANEL
            rect = plt.Rectangle((c, 9-r), 0.88, 0.88, fc=clr, ec=BG, linewidth=1.5, zorder=2)
            ax.add_patch(rect)
    ax.set_xlim(-0.1, 10); ax.set_ylim(-0.1, 10)
    ax.set_aspect('equal'); ax.axis('off')
    handles = [mpatches.Patch(color=colors[i],
               label=f'{"★ " if i==best_i else ""}{names[i]}  {scores[i]:.1f}%')
               for i in range(len(names))]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5,-0.22),
              ncol=2, fontsize=8.5, framealpha=0, labelcolor=TEXT)
    ax.set_title(f'🧇  OUTPUT FEATURE : {col.upper()}', color=TEXT, fontsize=12, pad=10, fontweight='bold')

# ─────────────────────────────────────────────────────────────────────────────
# CHART TYPE 10 — DOT PLOT (Cleveland)
# ─────────────────────────────────────────────────────────────────────────────
def chart_dot_plot(ax, names, scores, colors, col, best_i):
    y      = np.arange(len(names))
    mean_s = np.mean(scores)
    ax.axvline(mean_s, color=SUB, linewidth=1.2, linestyle='--', alpha=0.6, zorder=1)
    ax.text(mean_s + 0.5, len(names)-0.3, f'μ = {mean_s:.1f}%', fontsize=8, color=SUB, va='top')
    for i, (yi, sc, c) in enumerate(zip(y, scores, colors)):
        # Range bar from mean
        ax.plot([mean_s, sc], [yi, yi], color=c, linewidth=1.5, alpha=0.5, zorder=1)
        ms = 130 if i == best_i else 80
        ec = '#FFD700' if i == best_i else 'none'
        ax.scatter([sc], [yi], color=c, s=ms, zorder=3, edgecolors=ec, linewidth=2)
        ax.text(sc + 0.8, yi, f'{sc:.1f}%', va='center', fontsize=9.5,
                color=TEXT, fontweight='bold' if i == best_i else 'normal')
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlabel('Score (%)', fontsize=9); ax.set_xlim(0, 118)
    ax.set_title(f'•  OUTPUT FEATURE : {col.upper()}', color=TEXT, fontsize=12, pad=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.3); _best_tag(ax, names[best_i], scores[best_i])

# ─────────────────────────────────────────────────────────────────────────────
# CHART TYPE 11 — SMOOTH AREA
# ─────────────────────────────────────────────────────────────────────────────
def chart_area(ax, names, scores, colors, col, best_i):
    x = np.arange(len(names))
    if len(x) >= 4:
        xs  = np.linspace(0, len(names)-1, 300)
        spl = make_interp_spline(x, scores, k=3)
        ys  = np.clip(spl(xs), 0, 115)
    else:
        xs, ys = x.astype(float), np.array(scores)
    ax.fill_between(xs, ys, alpha=0.28, color=PALETTE[1])
    ax.plot(xs, ys, color=PALETTE[1], linewidth=2.5)
    for i, (xi, sc) in enumerate(zip(x, scores)):
        ms = 110 if i == best_i else 60
        ec = '#FFD700' if i == best_i else 'none'
        ax.scatter([xi], [sc], color=colors[i], s=ms, zorder=4, edgecolors=ec, linewidth=2)
        ax.text(xi, sc + 2.8, f'{sc:.1f}%', ha='center', fontsize=9,
                color=TEXT, fontweight='bold' if i == best_i else 'normal')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5, rotation=20, ha='right')
    ax.set_ylabel('Score (%)', fontsize=9); ax.set_ylim(0, 118)
    ax.set_title(f'📈  OUTPUT FEATURE : {col.upper()}', color=TEXT, fontsize=12, pad=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3); _best_tag(ax, names[best_i], scores[best_i])

# ─────────────────────────────────────────────────────────────────────────────
# CHART TYPE 12 — DIVERGING BAR (delta from mean)
# ─────────────────────────────────────────────────────────────────────────────
def chart_diverging_bar(ax, names, scores, colors, col, best_i):
    mean_s = np.mean(scores)
    delta  = [s - mean_s for s in scores]
    y      = np.arange(len(names))
    d_col  = [PALETTE[1] if d >= 0 else PALETTE[2] for d in delta]
    bars   = ax.barh(y, delta, color=d_col, height=0.55, linewidth=0, zorder=2)
    for i, (bar, dv, sc) in enumerate(zip(bars, delta, scores)):
        ha = 'left' if dv >= 0 else 'right'
        off = 0.35 if dv >= 0 else -0.35
        ax.text(dv + off, i, f'{sc:.1f}%', va='center', ha=ha, fontsize=9.5,
                color=TEXT, fontweight='bold' if i == best_i else 'normal')
        if i == best_i:
            bar.set_edgecolor('#FFD700'); bar.set_linewidth(2)
    ax.axvline(0, color=TEXT, linewidth=1.2, alpha=0.5)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlabel(f'Δ from mean  ({mean_s:.1f}%)', fontsize=9)
    ax.set_title(f'↔️  OUTPUT FEATURE : {col.upper()}', color=TEXT, fontsize=12, pad=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.3); _best_tag(ax, names[best_i], scores[best_i])

# ─────────────────────────────────────────────────────────────────────────────
# CHART TYPE 13 — HEATMAP STRIP
# ─────────────────────────────────────────────────────────────────────────────
def chart_heatmap(ax, names, scores, colors, col, best_i):
    data = np.array(scores).reshape(1, -1)
    im   = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=9.5, rotation=25, ha='right', color=TEXT)
    ax.set_yticks([])
    for j, sc in enumerate(scores):
        ax.text(j, 0, f'{sc:.1f}%', ha='center', va='center',
                fontsize=11, color='black', fontweight='bold')
        if j == best_i:
            rect = plt.Rectangle((j-0.5, -0.5), 1, 1,
                                  fill=False, edgecolor='#FFD700', linewidth=2.5)
            ax.add_patch(rect)
    cb = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.42, fraction=0.06)
    cb.ax.tick_params(colors=SUB, labelsize=8)
    ax.set_title(f'🔥  OUTPUT FEATURE : {col.upper()}', color=TEXT, fontsize=12, pad=10, fontweight='bold')
    _best_tag(ax, names[best_i], scores[best_i])

# ─────────────────────────────────────────────────────────────────────────────
# CHART TYPE 14 — WATERFALL
# ─────────────────────────────────────────────────────────────────────────────
def chart_waterfall(ax, names, scores, colors, col, best_i):
    bottoms = [0] + list(scores[:-1])
    heights = [scores[0]] + [scores[i] - scores[i-1] for i in range(1, len(scores))]
    bar_c   = [colors[0]] + [PALETTE[1] if h >= 0 else PALETTE[2] for h in heights[1:]]
    ax.bar(np.arange(len(names)), heights, bottom=bottoms, color=bar_c,
           width=0.6, linewidth=0, zorder=2)
    for i in range(len(scores)-1):
        ax.plot([i+0.3, i+0.7], [scores[i], scores[i]],
                color=SUB, linewidth=1.2, linestyle='--', alpha=0.7)
    for i, sc in enumerate(scores):
        ax.text(i, sc + 1.8, f'{sc:.1f}%', ha='center', fontsize=9,
                color=TEXT, fontweight='bold' if i == best_i else 'normal')
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=8.5, rotation=22, ha='right')
    ax.set_ylabel('Score (%)', fontsize=9); ax.set_ylim(0, 118)
    ax.set_title(f'🌊  OUTPUT FEATURE : {col.upper()}', color=TEXT, fontsize=12, pad=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3); _best_tag(ax, names[best_i], scores[best_i])

# ─────────────────────────────────────────────────────────────────────────────
# CHART TYPE 15 — POLAR BAR
# ─────────────────────────────────────────────────────────────────────────────
def chart_polar_bar(ax, names, scores, colors, col, best_i):
    N      = len(names)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    width  = 2*np.pi / N * 0.8
    bars   = ax.bar(angles, scores, width=width, color=colors, alpha=0.82,
                    edgecolor=BG, linewidth=1.5, bottom=8)
    for i, (a, sc) in enumerate(zip(angles, scores)):
        fw = 'bold' if i == best_i else 'normal'
        ax.text(a, sc + 14, f'{sc:.0f}%', ha='center', va='center',
                fontsize=8.5, color=TEXT, fontweight=fw)
        if i == best_i:
            bars[i].set_edgecolor('#FFD700'); bars[i].set_linewidth(2.5)
    ax.set_xticks(angles); ax.set_xticklabels(names, fontsize=9, color=TEXT)
    ax.set_yticks([]); ax.set_ylim(0, 140); ax.set_facecolor(PANEL)
    ax.tick_params(colors=SUB)
    ax.set_title(f'🌡️  OUTPUT FEATURE : {col.upper()}', color=TEXT, fontsize=12, pad=22, fontweight='bold')

# ─────────────────────────────────────────────────────────────────────────────
# CHART TYPE 16 — VERTICAL BAR
# ─────────────────────────────────────────────────────────────────────────────
def chart_vertical_bar(ax, names, scores, colors, col, best_i):
    x    = np.arange(len(names))
    bars = ax.bar(x, scores, color=colors, width=0.58, linewidth=0, zorder=2)
    for i, (bar, sc) in enumerate(zip(bars, scores)):
        ax.text(i, sc + 1.5, f'{sc:.1f}%', ha='center', fontsize=9.5,
                color=TEXT, fontweight='bold' if i == best_i else 'normal')
        if i == best_i:
            bar.set_edgecolor('#FFD700'); bar.set_linewidth(2)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5, rotation=25, ha='right')
    ax.set_ylabel('Score (%)', fontsize=9); ax.set_ylim(0, 118)
    ax.set_title(f'📊  OUTPUT FEATURE : {col.upper()}', color=TEXT, fontsize=12, pad=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3); _best_tag(ax, names[best_i], scores[best_i])

# ─────────────────────────────────────────────────────────────────────────────
# CHART TYPE 17 — TREEMAP
# ─────────────────────────────────────────────────────────────────────────────
def chart_treemap(ax, names, scores, colors, col, best_i):
    try:
        labels = [f'{"★ " if i==best_i else ""}{n}\n{s:.1f}%'
                  for i, (n, s) in enumerate(zip(names, scores))]
        squarify.plot(sizes=scores, label=labels, color=colors, alpha=0.88, ax=ax,
                      text_kwargs={'fontsize': 9.5, 'color': BG, 'fontweight': 'bold'},
                      bar_kwargs={'edgecolor': BG, 'linewidth': 2})
        ax.axis('off')
    except Exception:
        chart_vertical_bar(ax, names, scores, colors, col, best_i)
        return
    ax.set_title(f'🗺️  OUTPUT FEATURE : {col.upper()}', color=TEXT, fontsize=12, pad=10, fontweight='bold')

# ─────────────────────────────────────────────────────────────────────────────
# CHART TYPE 18 — STACKED NORMALIZED (100 % bar)
# ─────────────────────────────────────────────────────────────────────────────
def chart_stacked_norm(ax, names, scores, colors, col, best_i):
    total = sum(scores)
    fracs = [s / total for s in scores]
    left  = 0
    for i, (nm, fr, c) in enumerate(zip(names, fracs, colors)):
        lw = 2.5 if i == best_i else 0.5
        ec = '#FFD700' if i == best_i else BG
        ax.barh(0, fr, left=left, color=c, height=0.45,
                linewidth=lw, edgecolor=ec, zorder=2)
        if fr > 0.055:
            label = f'★ {nm}\n{scores[i]:.1f}%' if i == best_i else f'{nm}\n{scores[i]:.1f}%'
            ax.text(left + fr/2, 0, label, ha='center', va='center',
                    fontsize=8.5, color=BG, fontweight='bold')
        left += fr
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.5); ax.axis('off')
    ax.set_title(f'📏  OUTPUT FEATURE : {col.upper()}', color=TEXT, fontsize=12, pad=10, fontweight='bold')
    handles = [mpatches.Patch(color=colors[i], label=f'{names[i]} {scores[i]:.1f}%')
               for i in range(len(names))]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.55),
              ncol=3, fontsize=8.5, framealpha=0, labelcolor=TEXT)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — best annotation
# ─────────────────────────────────────────────────────────────────────────────
def _best_tag(ax, best_name, best_score, pie=False):
    if pie:
        ax.text(0, -1.22, f'★ Best: {best_name}  {best_score:.1f}%',
                ha='center', fontsize=8.5, color='#FFD700', fontweight='bold',
                transform=ax.transData)
    else:
        ax.text(0.99, 0.99, f'★ {best_name}  {best_score:.1f}%',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, color='#FFD700', fontweight='bold',
                bbox=dict(fc=PANEL, ec='#FFD700', boxstyle='round,pad=0.3', lw=1.2))


# ─────────────────────────────────────────────────────────────────────────────
# CHART TYPE ASSIGNMENT LIST  (cycles through 18 types)
# ─────────────────────────────────────────────────────────────────────────────
CHART_FUNCS = [
    chart_horizontal_bar,   # 0
    chart_pie,              # 1
    chart_radar,            # 2  ← needs polar projection
    chart_lollipop,         # 3
    chart_donut,            # 4
    chart_step_line,        # 5
    chart_bubble,           # 6
    chart_funnel,           # 7
    chart_waffle,           # 8
    chart_dot_plot,         # 9
    chart_area,             # 10
    chart_diverging_bar,    # 11
    chart_heatmap,          # 12
    chart_waterfall,        # 13
    chart_polar_bar,        # 14  ← needs polar projection
    chart_vertical_bar,     # 15
    chart_treemap,          # 16
    chart_stacked_norm,     # 17
]

CHART_NAMES = [
    'Horizontal Bar', 'Pie Chart', 'Radar/Spider', 'Lollipop Chart',
    'Donut Chart', 'Step Line', 'Bubble Chart', 'Funnel Chart',
    'Waffle Chart', 'Dot Plot (Cleveland)', 'Smooth Area', 'Diverging Bar',
    'Heatmap Strip', 'Waterfall', 'Polar Bar', 'Vertical Bar',
    'Treemap', 'Stacked 100% Bar',
]

POLAR_CHARTS = {chart_radar, chart_polar_bar}


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE INDIVIDUAL COLUMN CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("  GENERATING COLUMN-LEVEL VISUALIZATIONS  (18 chart types)")
print("="*80)

ordered_cols = list(results_store.keys())

for idx, col in enumerate(ordered_cols):
    names, scores, colors, best_i, task_ = get_data(col)
    chart_fn   = CHART_FUNCS[idx % len(CHART_FUNCS)]
    chart_name = CHART_NAMES[idx % len(CHART_NAMES)]

    if chart_fn in POLAR_CHARTS:
        fig, ax = plt.subplots(figsize=(7, 6.5), subplot_kw={'projection': 'polar'})
    else:
        fig, ax = plt.subplots(figsize=(9.5, 5.5))

    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    chart_fn(ax, names, scores, colors, col, best_i)

    fig.text(0.5, 0.01,
             f'Chart Type: {chart_name}   |   Task: {task_.upper()}   |   Col {idx+1}/{len(ordered_cols)}',
             ha='center', fontsize=8, color=SUB, style='italic')

    plt.tight_layout(pad=1.8)
    plt.savefig(f'col_{idx+1:02d}_{col}.png', dpi=130,
                bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.show()
    print(f"  ✓  Col {idx+1:>2}/{len(ordered_cols)}  [{chart_name:<28}]  →  {col}")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL COMPARATIVE CHART
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating FINAL COMPARATIVE CHART ...")

all_model_names = []
for col in ordered_cols:
    R, _ = results_store[col]
    for r in R:
        nm = MODEL_SHORT.get(r[0], r[0])
        if nm not in all_model_names:
            all_model_names.append(nm)

mc_final = [MODEL_COLORS_MAP.get(m, PALETTE[i % len(PALETTE)])
            for i, m in enumerate(all_model_names)]

n_models = len(all_model_names)
n_cols   = len(ordered_cols)

score_matrix = np.full((n_models, n_cols), np.nan)
for j, col in enumerate(ordered_cols):
    R, _ = results_store[col]
    for r in R:
        nm = MODEL_SHORT.get(r[0], r[0])
        if nm in all_model_names:
            i = all_model_names.index(nm)
            score_matrix[i, j] = r[1] * 100

# ── Layout: main grouped bar + best-per-column strip ─────────────────────────
fig = plt.figure(figsize=(max(20, n_cols * 1.15), 15), facecolor=BG)
gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)

ax_main = fig.add_subplot(gs[0]); ax_main.set_facecolor(PANEL)
ax_bot  = fig.add_subplot(gs[1]); ax_bot.set_facecolor(PANEL)

group_w  = 0.8
bar_w    = group_w / n_models
x        = np.arange(n_cols)

for i, (mdl, mc) in enumerate(zip(all_model_names, mc_final)):
    offset = (i - n_models / 2 + 0.5) * bar_w
    vals   = score_matrix[i]
    safe   = np.where(np.isnan(vals), 0, vals)
    ax_main.bar(x + offset, safe, bar_w * 0.88, color=mc, alpha=0.85,
                label=mdl, linewidth=0, zorder=2)

# Gold star above best bar per column
for j in range(n_cols):
    col_sc = score_matrix[:, j]
    if np.all(np.isnan(col_sc)): continue
    bi     = int(np.nanargmax(col_sc))
    bscore = col_sc[bi]
    offset = (bi - n_models / 2 + 0.5) * bar_w
    ax_main.scatter(j + offset, bscore + 1.8, marker='*', s=90,
                    color='#FFD700', zorder=6)

ax_main.set_xticks(x)
ax_main.set_xticklabels([c.upper() for c in ordered_cols],
                         fontsize=8, rotation=42, ha='right', color=TEXT)
ax_main.set_ylabel('Score (%)', fontsize=11, color=TEXT)
ax_main.set_ylim(0, 115)
ax_main.set_title('MODEL PERFORMANCE COMPARISON — ALL TARGET COLUMNS',
                   color=TEXT, fontsize=14, fontweight='bold', pad=14)
ax_main.legend(loc='upper right', fontsize=9, framealpha=0.18,
               labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID_C)
ax_main.grid(axis='y', alpha=0.3, zorder=0)
ax_main.tick_params(colors=SUB)

# ── Bottom strip: best score & model per column ───────────────────────────────
best_scores = []
best_models = []
best_colors_b = []
for j in range(n_cols):
    cs = score_matrix[:, j]
    if np.all(np.isnan(cs)):
        best_scores.append(0); best_models.append('N/A'); best_colors_b.append(PANEL)
    else:
        bi = int(np.nanargmax(cs))
        best_scores.append(float(cs[bi]))
        best_models.append(all_model_names[bi])
        best_colors_b.append(mc_final[bi])

ax_bot.bar(x, best_scores, 0.62, color=best_colors_b, alpha=0.9, linewidth=0, zorder=2)
for j, (bs, bm) in enumerate(zip(best_scores, best_models)):
    ax_bot.text(j, bs + 1.2, f'{bs:.1f}%', ha='center', fontsize=7.5, color=TEXT, fontweight='bold')
    ax_bot.text(j, -7, bm, ha='center', fontsize=6.8, color=SUB, rotation=32, va='top')

ax_bot.set_xticks(x); ax_bot.set_xticklabels(['' for _ in ordered_cols])
ax_bot.set_ylabel('Best Score', fontsize=9, color=TEXT)
ax_bot.set_ylim(-12, 115)
ax_bot.set_title('★  Best Model per Column', color=TEXT, fontsize=10,
                  fontweight='bold', pad=6)
ax_bot.grid(axis='y', alpha=0.3, zorder=0)
ax_bot.tick_params(colors=SUB)

# ── Stats banner ──────────────────────────────────────────────────────────────
avg_acc = np.mean([s[3] for s in summary]) * 100
best_entry    = max(summary, key=lambda x: x[3])
best_ov_col   = best_entry[0]
best_ov_model = best_entry[2]
best_ov_score = best_entry[3] * 100

# Overall best model across all columns (most frequent best)
from collections import Counter
best_model_counts = Counter([s[2] for s in summary])
overall_best_model, cnt_occ = best_model_counts.most_common(1)[0]

banner = (f'   ⌀ Avg Accuracy (best/col): {avg_acc:.2f}%'
          f'     |     🏆 Highest Single Score: {best_ov_model} on {best_ov_col.upper()} → {best_ov_score:.2f}%'
          f'     |     👑 Most Frequent Best Model: {overall_best_model} ({cnt_occ}/{len(summary)} cols)   ')

fig.text(0.5, 0.002, banner, ha='center', fontsize=10, color='#FFD700', fontweight='bold',
         bbox=dict(facecolor=PANEL, edgecolor='#FFD700', boxstyle='round,pad=0.5', linewidth=1.8))

plt.savefig('final_comparative_chart.png', dpi=150,
            bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.show()

# ── Console summary ───────────────────────────────────────────────────────────
print("\n" + "="*80)
print("  VISUALIZATION COMPLETE")
print("="*80)
print(f"  • {len(ordered_cols)} column charts saved  →  col_01_*.png … col_{len(ordered_cols):02d}_*.png")
print(f"  • Comparative chart saved  →  final_comparative_chart.png")
print(f"\n  ⌀  Average Accuracy          : {avg_acc:.2f}%")
print(f"  🏆 Highest Single Score      : {best_ov_model}  on  {best_ov_col.upper()}  →  {best_ov_score:.2f}%")
print(f"  👑 Most Frequent Best Model  : {overall_best_model}  ({cnt_occ}/{len(summary)} columns)")
print("="*80)