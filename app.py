import json
import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Ekonomi Indonesia",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CSS
# =========================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Mono:wght@400;500;700&display=swap');

:root {
  --bg:        #0b0f19;
  --surface:   #111827;
  --surface2:  #1a2235;
  --border:    rgba(255,255,255,0.07);
  --text:      #e8edf5;
  --muted:     #6b7a99;
  --accent:    #22d3a4;
  --accent2:   #3b82f6;
  --danger:    #f87171;
  --warn:      #fbbf24;
  --purple:    #a78bfa;
  --orange:    #fb923c;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }
.main .block-container { padding: 1.2rem 1.8rem 2.5rem; max-width: 1480px; }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

.hero {
    background: linear-gradient(135deg, #0f2027 0%, #0b3d2e 50%, #0d2147 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 26px 30px;
    margin-bottom: 18px;
    position: relative; overflow: hidden;
}
.hero::after {
    content:''; position: absolute;
    bottom: -50px; right: -50px;
    width: 220px; height: 220px; border-radius: 50%;
    background: radial-gradient(circle, rgba(34,211,164,.12), transparent 70%);
    pointer-events: none;
}
.hero h1 { font-size: 26px; font-weight: 700; margin: 0 0 4px; color: #fff; }
.hero p  { font-size: 13px; margin: 0; color: rgba(255,255,255,.58); }

.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px 18px 15px;
    position: relative; overflow: hidden;
    transition: transform .15s ease, border-color .15s ease;
    min-height: 92px;
}
.kpi-card:hover { transform: translateY(-2px); border-color: rgba(34,211,164,.3); }
.kpi-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: var(--card-accent, var(--accent));
    border-radius: 12px 12px 0 0;
}
.kpi-label {
    font-size: 10px; font-weight: 700;
    letter-spacing: 1.2px; text-transform: uppercase;
    color: var(--muted); margin-bottom: 10px;
}
.kpi-value {
    font-size: 20px; font-weight: 700; line-height: 1;
    margin-bottom: 6px; color: var(--text);
    font-family: 'DM Mono', monospace;
}
.kpi-sub { font-size: 11px; color: var(--muted); }
.kpi-badge {
    display: inline-block; padding: 2px 7px;
    border-radius: 99px; font-size: 11px; font-weight: 700; margin-left: 4px;
}
.badge-up   { background: rgba(34,211,164,.15); color: #22d3a4; }
.badge-down { background: rgba(248,113,113,.15); color: #f87171; }
.badge-flat { background: rgba(59,130,246,.15); color: #60a5fa; }

.sec-head {
    display: flex; align-items: center; gap: 10px;
    margin: 4px 0 14px;
}
.sec-head-bar {
    width: 3px; height: 18px; border-radius: 3px;
    background: var(--accent); flex-shrink: 0;
}
.sec-head-text { font-size: 13px; font-weight: 700; color: var(--text); }
.sec-head-badge { margin-left: auto; font-size: 10px; color: var(--muted); font-family: 'DM Mono', monospace; }

.insight-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 14px;
    padding: 14px 16px;
    margin: 8px 0 16px;
}
.insight-box.warn { border-left-color: var(--warn); }
.insight-box.bad  { border-left-color: var(--danger); }
.insight-box.good { border-left-color: var(--accent); }
.ins-title {
    font-size: 11px; font-weight: 700; letter-spacing: 1.2px;
    text-transform: uppercase; color: var(--muted); margin-bottom: 8px;
}
.ins-list { margin: 0; padding-left: 18px; color: var(--text); font-size: 13px; line-height: 1.55; }

.filter-bar {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px; margin-bottom: 18px;
}

.chip {
    display: inline-flex; align-items: center; gap: 5px;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 99px; padding: 4px 12px;
    font-size: 11px; color: var(--muted); margin: 2px 2px 8px;
}

.sb-section {
    font-size: 10px; font-weight: 700; letter-spacing: 1.4px;
    text-transform: uppercase; color: var(--muted);
    margin: 16px 0 8px; padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}

[data-testid="stMetric"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 12px 14px;
}
[data-testid="stMetricLabel"] { font-size: 11px !important; color: var(--muted) !important; }
[data-testid="stMetricValue"] { font-size: 22px !important; font-family: 'DM Mono', monospace; }

[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
.js-plotly-plot .plotly { border-radius: 12px; overflow: hidden; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--surface2); border-radius: 4px; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# PLOTLY THEME
# =========================================================
PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(17,24,39,0)",
    plot_bgcolor="rgba(17,24,39,0)",
    font=dict(family="DM Sans", color="#9aa5be", size=11),
    margin=dict(t=16, b=16, l=8, r=8),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.07)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.07)"),
    hoverlabel=dict(bgcolor="#1a2235", bordercolor="rgba(255,255,255,.12)", font=dict(family="DM Sans", size=12, color="#e8edf5")),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", font=dict(size=11))
)

COLORS = {
    "green": "#22d3a4", "blue": "#3b82f6", "red": "#f87171",
    "yellow": "#fbbf24", "purple": "#a78bfa", "orange": "#fb923c",
    "pink": "#f472b6", "cyan": "#22d3ee", "warn": "#fbbf24",
}
PROV_PALETTE = ["#22d3a4", "#3b82f6", "#f87171", "#fbbf24", "#a78bfa",
                "#fb923c", "#34d399", "#60a5fa", "#f472b6", "#4ade80"]

# =========================================================
# DATA LOADING
# =========================================================
DATA_PATH = "Data/Indonesia Dashboard Data Clean.xlsx"

#@st.cache_data(show_spinner=False)
def load_data():
    gini     = pd.read_excel(DATA_PATH, sheet_name="Gini_Ratio")
    inflasi  = pd.read_excel(DATA_PATH, sheet_name="Inflasi")
    neraca   = pd.read_excel(DATA_PATH, sheet_name="Neraca_Perdagangan")
    pdrb     = pd.read_excel(DATA_PATH, sheet_name="PDRB_PerKapita")
    penduduk = pd.read_excel(DATA_PATH, sheet_name="Penduduk")
    tpt      = pd.read_excel(DATA_PATH, sheet_name="Pengangguran_TPT")
    miskin   = pd.read_excel(DATA_PATH, sheet_name="Kemiskinan")

    num_cols_mapping = [
        (pdrb, ["PDRB_PerKapita_RibuRupiah"]),
        (tpt, ["TPT_Persen"]),
        (miskin, ["Persen_Penduduk_Miskin"]),
        (gini, ["Gini_Ratio"]),
        (inflasi, ["Inflasi_YoY_Persen"]),
        (penduduk, ["Jumlah_Penduduk_Ribu", "Kepadatan_per_Km2", "Laju_Pertumbuhan_Persen",
                    "Persentase_Penduduk", "Rasio_Jenis_Kelamin"]),
    ]

    for df, cols in num_cols_mapping:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    for col in ["Ekspor_Migas", "Ekspor_NonMigas", "Impor_Migas", "Impor_NonMigas"]:
        if col in neraca.columns:
            neraca[col] = pd.to_numeric(neraca[col], errors="coerce").fillna(0)
        else:
            neraca[col] = 0

    neraca["Total_Ekspor"] = neraca["Ekspor_Migas"] + neraca["Ekspor_NonMigas"]
    neraca["Total_Impor"]  = neraca["Impor_Migas"]  + neraca["Impor_NonMigas"]
    neraca["Net_Trade"]    = neraca["Total_Ekspor"]  - neraca["Total_Impor"]

    for df in [gini, inflasi, neraca, pdrb, penduduk, tpt, miskin]:
        if "Provinsi" in df.columns:
            df["Provinsi"] = df["Provinsi"].astype(str).str.strip()
        if "Tahun" in df.columns:
            df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce").astype("Int64")

    return gini, inflasi, neraca, pdrb, penduduk, tpt, miskin

@st.cache_data(show_spinner=False)
def load_geojsons():
    with open("Maps/prov 34_fixed.geojson", encoding="utf-8") as f:
        g34 = json.load(f)

    with open("Maps/38 Provinsi Indonesia - Provinsi.json", encoding="utf-8") as f:
        g38 = json.load(f)

    return g34, g38

with st.spinner("Memuat data…"):
    gini, inflasi, neraca, pdrb, penduduk, tpt, miskin = load_data()
    geojson_34, geojson_38 = load_geojsons()

# =========================================================
# CONSTANTS
# =========================================================
NAME_MAP = {
    "ACEH":"Aceh","SUMATERA UTARA":"Sumatera Utara","SUMATERA BARAT":"Sumatera Barat",
    "RIAU":"Riau","KEP. RIAU":"Kepulauan Riau","KEPULAUAN RIAU":"Kepulauan Riau",
    "JAMBI":"Jambi","SUMATERA SELATAN":"Sumatera Selatan",
    "KEP. BANGKA BELITUNG":"Kepulauan Bangka Belitung","BANGKA BELITUNG":"Kepulauan Bangka Belitung",
    "BENGKULU":"Bengkulu","LAMPUNG":"Lampung","DKI JAKARTA":"DKI Jakarta",
    "JAWA BARAT":"Jawa Barat","JAWA TENGAH":"Jawa Tengah","DI YOGYAKARTA":"DI Yogyakarta",
    "D.I. YOGYAKARTA":"DI Yogyakarta","JAWA TIMUR":"Jawa Timur","BANTEN":"Banten","BALI":"Bali",
    "NUSA TENGGARA BARAT":"Nusa Tenggara Barat","NUSA TENGGARA TIMUR":"Nusa Tenggara Timur",
    "KALIMANTAN BARAT":"Kalimantan Barat","KALIMANTAN TENGAH":"Kalimantan Tengah",
    "KALIMANTAN SELATAN":"Kalimantan Selatan","KALIMANTAN TIMUR":"Kalimantan Timur",
    "KALIMANTAN UTARA":"Kalimantan Utara","SULAWESI UTARA":"Sulawesi Utara",
    "GORONTALO":"Gorontalo","SULAWESI TENGAH":"Sulawesi Tengah","SULAWESI SELATAN":"Sulawesi Selatan",
    "SULAWESI BARAT":"Sulawesi Barat","SULAWESI TENGGARA":"Sulawesi Tenggara",
    "MALUKU":"Maluku","MALUKU UTARA":"Maluku Utara","PAPUA BARAT":"Papua Barat",
    "PAPUA BARAT DAYA":"Papua Barat Daya","PAPUA":"Papua","PAPUA SELATAN":"Papua Selatan",
    "PAPUA TENGAH":"Papua Tengah","PAPUA PEGUNUNGAN":"Papua Pegunungan",
    "NANGROE ACEH DARUSALAM":"Aceh","INDONESIA":"Indonesia",
}
PROV_BARU = {"PAPUA BARAT DAYA", "PAPUA SELATAN", "PAPUA TENGAH", "PAPUA PEGUNUNGAN"}

# =========================================================
# SESSION STATE
# =========================================================
defaults = {
    "active_tab": "summary",
    "bookmarks": [],
    "panel_open": True,
    "compare_mode": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================================================
# HELPERS
# =========================================================
def sec(title: str, note: str = "", color: str = "var(--accent)") -> None:
    """Render a section header with an optional note."""
    badge = f'<span class="sec-head-badge">{note}</span>' if note else ""
    st.markdown(
        f'<div class="sec-head"><div class="sec-head-bar" style="background:{color}"></div>'
        f'<div class="sec-head-text">{title}</div>{badge}</div>',
        unsafe_allow_html=True,
    )


def apply_layout(fig, h: int = 320, legend_h: bool = False, **kwargs):
    """Apply a consistent dark layout to plotly figures."""
    layout = {**PLOT_LAYOUT, "height": h, **kwargs}
    if legend_h:
        layout["legend"] = {
            **PLOT_LAYOUT.get("legend", {}),
            "orientation": "h",
            "y": -0.22,
            "x": 0,
            "font": {"size": 10},
        }
    fig.update_layout(**layout)
    return fig

def format_delta(
    val: Optional[float],
    suffix: str = "",
    reverse: bool = False,
    fmt: str = ".2f",
) -> str:
    """
    Return an HTML string for a delta value with arrow, sign, and colour.
    Example: ▲ +6.15%  (green) or ▼ -2.15% (red).
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    arrow = "▲" if val > 0 else ("▼" if val < 0 else "•")
    sign = "+" if val > 0 else "-"
    color = "green" if (val > 0) ^ reverse else "red"
    return f'<span style="color:{color}; font-weight:600;">{arrow} {sign}{abs(val):{fmt}}{suffix}</span>'


def kpi(
    label: str,
    value: str,
    sub: str = "",
    delta: Optional[float] = None,
    delta_suffix: str = "",
    reverse: bool = False,
    color: str = "var(--accent)",
    delta_fmt: str = ".2f",
) -> str:
    """Create a KPI card with an optional delta (YoY change)."""
    delta_html = (
        format_delta(delta, delta_suffix, reverse, delta_fmt) if delta is not None else ""
    )
    return f"""<div class="kpi-card" style="--card-accent:{color}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub} {delta_html}</div>
    </div>"""

def insight_callout(title: str, bullets: List[str], tone: str = "info") -> None:
    """Render an insight box with bullet points."""
    if not bullets:
        return
    palette = {
        "info": COLORS["blue"],
        "good": COLORS["green"],
        "warn": COLORS["warn"],
        "bad": COLORS["red"],
    }
    color = palette.get(tone, COLORS["blue"])
    items = "".join(f"<li style='margin-bottom:4px'>{b}</li>" for b in bullets)
    st.markdown(
        f"""
    <div class="insight-box" style="border-left-color:{color}">
        <div class="ins-title">💡 {title}</div>
        <ul class="ins-list">{items}</ul>
    </div>""",
        unsafe_allow_html=True,
    )

def fmt_v(v, digits=2, suffix=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:,.{digits}f}{suffix}"

def format_rupiah_auto(v, digits=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"

    v = float(v)
    sign = "-" if v < 0 else ""
    v = abs(v)

    if v >= 1_000_000_000:
        return f"{sign}Rp {v/1_000_000_000:,.{digits}f} Miliar"
    elif v >= 1_000_000:
        return f"{sign}Rp {v/1_000_000:,.{digits}f} Juta"
    elif v >= 1_000:
        return f"{sign}Rp {v/1_000:,.{digits}f} Ribu"
    else:
        return f"{sign}Rp {v:,.{digits}f}"


def format_delta_display(v, unit="", digits=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"

    if unit == "Rp":
        sign = "+" if v >= 0 else "-"
        return f"{sign}{format_rupiah_auto(abs(v), digits)}"

    if unit == "%":
        return f"{v:+.{digits}f}%"

    if unit:
        return f"{v:+.{digits}f} {unit}"

    return f"{v:+.{digits}f}"

def clean_label(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return str(x)
    return NAME_MAP.get(str(x).strip().upper(), str(x).strip().title())


def non_country(df: pd.DataFrame) -> pd.DataFrame:
    if "Provinsi" not in df.columns:
        return df.copy()
    return df[~df["Provinsi"].astype(str).str.upper().isin({"INDONESIA"})].copy()


def country_only(df: pd.DataFrame) -> pd.DataFrame:
    if "Provinsi" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["Provinsi"].astype(str).str.upper().isin({"INDONESIA", "INDONESIA"})].copy()


def yr_filter(df: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    return df[(df["Tahun"] >= y0) & (df["Tahun"] <= y1)].copy()


def years_of(df: pd.DataFrame) -> List[int]:
    return sorted([int(v) for v in df["Tahun"].dropna().unique()])


def get_prov_list(df: pd.DataFrame) -> List[str]:
    return sorted(non_country(df)["Provinsi"].dropna().astype(str).unique().tolist())


def map_indicator(indicator: str):
    """Returns (df[Provinsi,Tahun,value], title, colorscale, unit)"""
    if indicator == "PDRB/Kapita":
        df = non_country(pdrb)[["Provinsi", "Tahun", "PDRB_PerKapita_RibuRupiah"]].copy()
        df.rename(columns={"PDRB_PerKapita_RibuRupiah": "value"}, inplace=True)

        # dari ribu rupiah -> rupiah penuh
        df["value"] = df["value"] * 1000

        return df, "PDRB/Kapita", "Teal", "Rp"

    if indicator == "Pengangguran (TPT)":
        df = tpt[
            (tpt["Periode"] == "Agustus") &
            (tpt["Provinsi"].str.upper() != "INDONESIA")
        ][["Provinsi", "Tahun", "TPT_Persen"]].copy()
        df.rename(columns={"TPT_Persen": "value"}, inplace=True)
        return df, "TPT (%)", "Reds", "%"

    if indicator == "Kemiskinan":
        df = miskin[
            (miskin["Daerah"] == "Jumlah") &
            (miskin["Semester"] == "Semester 1 (Maret)") &
            (miskin["Provinsi"].str.upper() != "INDONESIA")
        ][["Provinsi", "Tahun", "Persen_Penduduk_Miskin"]].copy()
        df.rename(columns={"Persen_Penduduk_Miskin": "value"}, inplace=True)
        return df, "Kemiskinan (%)", "Purp", "%"

    if indicator == "Gini Ratio":
        df = gini[
            (gini["Daerah"] == "Perkotaan+Perdesaan") &
            (gini["Semester"] == "Semester 1 (Maret)") &
            (gini["Provinsi"].str.upper() != "INDONESIA")
        ][["Provinsi", "Tahun", "Gini_Ratio"]].copy()
        df.rename(columns={"Gini_Ratio": "value"}, inplace=True)
        return df, "Gini Ratio", "YlOrBr", ""

    df = inflasi[inflasi["Provinsi"].str.upper() != "INDONESIA"] \
        .groupby(["Provinsi", "Tahun"], as_index=False)["Inflasi_YoY_Persen"].mean()
    df.rename(columns={"Inflasi_YoY_Persen": "value"}, inplace=True)
    return df, "Inflasi YoY (%)", "OrRd", "%"

def render_bookmarks():
    if not st.session_state.bookmarks:
        st.caption("Belum ada bookmark.")
        return
    for b in st.session_state.bookmarks:
        c1, _, c3 = st.columns([5, 1, 1])
        with c1:
            st.write(f"**{b.get('name','—')}**")
            st.caption(f"Tab: {b.get('tab','')}")
        with c3:
            if st.button("Load", key=f"bm_{b.get('name')}"):
                st.session_state.active_tab = b.get("tab", "summary")
                st.rerun()


def normalize_series(s: pd.Series, invert: bool = False) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        out = pd.Series([50.0] * len(s), index=s.index)
    else:
        out = (s - mn) / (mx - mn) * 100
    if invert:
        out = 100 - out
    return out


def safe_get_metric(df: pd.DataFrame, prov: str, yr: int, col: str) -> float:
    if df is None or df.empty:
        return np.nan
    if "Provinsi" not in df.columns or "Tahun" not in df.columns or col not in df.columns:
        return np.nan
    r = df[(df["Provinsi"] == prov) & (df["Tahun"] == yr)]
    if r.empty:
        return np.nan
    ser = r[col].dropna()
    return float(ser.iloc[0]) if not ser.empty else np.nan


def top_and_bottom_text(df: pd.DataFrame, value_col: str, label_col: str = "provinsi_name"):
    if df.empty:
        return [], []
    top = df.loc[df[value_col].idxmax(), label_col]
    bot = df.loc[df[value_col].idxmin(), label_col]
    return top, bot

INDICATOR_CONFIG = {
    "PDRB/Kapita (Rp Ribu)": {"sheet": "pdrb", "ycol": "PDRB_PerKapita_RibuRupiah", "mode": "yearly", "title": "PDRB/Kapita (Rp Ribu)", "invert": False},
    "TPT (%)": {"sheet": "tpt", "ycol": "TPT_Persen", "mode": "yearly_filter", "filter_col": "Periode", "filter_val": "Agustus", "title": "TPT (%)", "invert": True},
    "Kemiskinan (%)": {"sheet": "miskin", "ycol": "Persen_Penduduk_Miskin", "mode": "yearly_filter", "filter_col_1": "Daerah", "filter_val_1": "Jumlah", "filter_col_2": "Semester", "filter_val_2": "Semester 1 (Maret)", "title": "Kemiskinan (%)", "invert": True},
    "Gini Ratio": {"sheet": "gini", "ycol": "Gini_Ratio", "mode": "yearly_filter", "filter_col_1": "Daerah", "filter_val_1": "Perkotaan+Perdesaan", "filter_col_2": "Semester", "filter_val_2": "Semester 1 (Maret)", "title": "Gini Ratio", "invert": True},
    "Inflasi YoY (%)": {"sheet": "inflasi", "ycol": "Inflasi_YoY_Persen", "mode": "yearly_mean", "title": "Inflasi YoY (%)", "invert": True},
    "Penduduk - Jumlah (Ribu)": {"sheet": "penduduk", "ycol": "Jumlah_Penduduk_Ribu", "mode": "yearly", "title": "Penduduk - Jumlah (Ribu)", "invert": False},
    "Penduduk - Kepadatan per Km2": {"sheet": "penduduk", "ycol": "Kepadatan_per_Km2", "mode": "yearly", "title": "Penduduk - Kepadatan per Km2", "invert": False},
    "Penduduk - Laju Pertumbuhan (%)": {"sheet": "penduduk", "ycol": "Laju_Pertumbuhan_Persen", "mode": "yearly", "title": "Penduduk - Laju Pertumbuhan (%)", "invert": False},
    "Penduduk - Persentase (%)": {"sheet": "penduduk", "ycol": "Persentase_Penduduk", "mode": "yearly", "title": "Penduduk - Persentase (%)", "invert": False},
    "Penduduk - Rasio Jenis Kelamin": {"sheet": "penduduk", "ycol": "Rasio_Jenis_Kelamin", "mode": "yearly", "title": "Penduduk - Rasio Jenis Kelamin", "invert": False},
    "Neraca - Total Ekspor": {"sheet": "neraca", "ycol": "Total_Ekspor", "mode": "yearly", "title": "Neraca - Total Ekspor", "invert": False},
    "Neraca - Total Impor": {"sheet": "neraca", "ycol": "Total_Impor", "mode": "yearly", "title": "Neraca - Total Impor", "invert": False},
    "Neraca - Net Trade": {"sheet": "neraca", "ycol": "Net_Trade", "mode": "yearly", "title": "Neraca - Net Trade", "invert": False},
    "Neraca - Ekspor Migas": {"sheet": "neraca", "ycol": "Ekspor_Migas", "mode": "yearly", "title": "Neraca - Ekspor Migas", "invert": False},
    "Neraca - Ekspor Nonmigas": {"sheet": "neraca", "ycol": "Ekspor_NonMigas", "mode": "yearly", "title": "Neraca - Ekspor Nonmigas", "invert": False},
    "Neraca - Impor Migas": {"sheet": "neraca", "ycol": "Impor_Migas", "mode": "yearly", "title": "Neraca - Impor Migas", "invert": False},
    "Neraca - Impor Nonmigas": {"sheet": "neraca", "ycol": "Impor_NonMigas", "mode": "yearly", "title": "Neraca - Impor Nonmigas", "invert": False},
}

def get_indicator_base_df(indicator_name: str) -> pd.DataFrame:
    cfg = INDICATOR_CONFIG[indicator_name]
    sheet = cfg["sheet"]
    ycol = cfg["ycol"]

    if sheet == "pdrb":
        return non_country(pdrb)[["Provinsi", "Tahun", ycol]].rename(columns={ycol: "value"}).copy()

    if sheet == "tpt":
        df = tpt[tpt["Periode"] == cfg["filter_val"]][["Provinsi", "Tahun", ycol]].copy()
        return non_country(df).rename(columns={ycol: "value"})

    if sheet == "miskin":
        df = miskin[(miskin[cfg["filter_col_1"]] == cfg["filter_val_1"]) & (miskin[cfg["filter_col_2"]] == cfg["filter_val_2"])][["Provinsi", "Tahun", ycol]].copy()
        return non_country(df).rename(columns={ycol: "value"})

    if sheet == "gini":
        df = gini[(gini[cfg["filter_col_1"]] == cfg["filter_val_1"]) & (gini[cfg["filter_col_2"]] == cfg["filter_val_2"])][["Provinsi", "Tahun", ycol]].copy()
        return non_country(df).rename(columns={ycol: "value"})

    if sheet == "inflasi":
        df = non_country(inflasi).groupby(["Provinsi", "Tahun"], as_index=False)[ycol].mean()
        return df.rename(columns={ycol: "value"})

    if sheet == "penduduk":
        cols = ["Provinsi", "Tahun", ycol]
        df = non_country(penduduk)[cols].copy()
        # Penduduk uses Title Case — normalize to UPPERCASE to match prov_list
        df["Provinsi"] = df["Provinsi"].astype(str).str.upper()
        return df.rename(columns={ycol: "value"})

    if sheet == "neraca":
        cols = ["Provinsi", "Tahun", ycol]
        df = non_country(neraca).copy()
        # Neraca uses variant names (e.g. "NANGROE ACEH DARUSALAM", "D.I. YOGYAKARTA")
        # Normalize via NAME_MAP then convert back to UPPERCASE to match prov_list
        df["Provinsi"] = (
            df["Provinsi"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map(lambda x: NAME_MAP.get(x, x).upper())
        )
        if ycol in df.columns:
            return df[cols].rename(columns={ycol: "value"}).copy()
        return pd.DataFrame(columns=["Provinsi", "Tahun", "value"])

    return pd.DataFrame(columns=["Provinsi", "Tahun", "value"])

def forecast_linear(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    df = df.dropna(subset=["Tahun", "value"]).copy()
    df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce")
    df = df.dropna(subset=["Tahun"])
    df["Tahun"] = df["Tahun"].astype(int)
    # Deduplicate: if multiple rows per year, take mean
    df = df.groupby("Tahun", as_index=False)["value"].mean()
    if len(df) < 2:
        return pd.DataFrame()

    X = df[["Tahun"]].values
    y = pd.to_numeric(df["value"], errors="coerce").values
    ok = ~np.isnan(y)
    X, y = X[ok], y[ok]
    if len(y) < 2:
        return pd.DataFrame()

    model = LinearRegression()
    model.fit(X, y)

    last_year = int(df["Tahun"].max())
    future_years = np.arange(last_year + 1, last_year + horizon + 1).reshape(-1, 1)
    pred_hist = model.predict(X)
    resid = y - pred_hist
    resid_std = float(np.std(resid)) if len(resid) > 1 else 0.0
    fut_pred = model.predict(future_years)

    out = pd.DataFrame({
        "Tahun": future_years.flatten(),
        "value": np.nan,
        "pred": fut_pred,
        "lower": fut_pred - 1.96 * resid_std,
        "upper": fut_pred + 1.96 * resid_std,
        "type": "Forecast"
    })
    hist = df.copy()
    hist["pred"] = hist["value"]
    hist["lower"] = np.nan
    hist["upper"] = np.nan
    hist["type"] = "Actual"
    return pd.concat([hist, out], ignore_index=True)

# =========================
# HERO
# =========================
st.markdown(
    """
<div class="hero">
  <h1>🇮🇩 Dashboard Ekonomi Indonesia</h1>
  <p>Analisis interaktif indikator makroekonomi · 38 provinsi · 2016–2026 · Data: BPS · Hover chart untuk tooltip detail</p>
</div>
""",
    unsafe_allow_html=True,
)

# =========================
# SIDEBAR
# =========================
with st.sidebar:

    # =========================
    # HEADER
    # =========================
    st.markdown(
        """
    <div style="padding:18px 4px 8px;text-align:center;">
      <div style="font-size:38px;line-height:1">🇮🇩</div>
      <div style="font-weight:700;font-size:16px;margin-top:8px;color:#e8edf5;">Ekonomi Indonesia</div>
      <div style="font-size:11px;color:#6b7a99;margin-top:2px;">BPS · 2016–2026</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # =========================
    # NAVIGASI (DROPDOWN)
    # =========================
    st.markdown('<div class="sb-section">Navigasi</div>', unsafe_allow_html=True)

    TAB_MAIN = [
        ("summary",    "📊 Ringkasan"),
        ("map",        "🗺️ Peta"),
        ("trend",      "📈 Tren"),
        ("comparison", "⚖️ Perbandingan"),
        ("trade",      "🌐 Neraca"),
        ("population", "👥 Penduduk"),
        ("forecast",   "🔮 Forecast"),
        ("ai",         "🤖 AI Analytics"),
        ("more",       "⋯ Lainnya"),
    ]

    TAB_LABELS = dict(TAB_MAIN)

    tab_keys   = [k for k, _ in TAB_MAIN]
    tab_labels = [v for _, v in TAB_MAIN]

    key_to_label = dict(TAB_MAIN)
    label_to_key = {v: k for k, v in TAB_MAIN}

    # default tab
    default_idx = 0
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = tab_keys[0]

    if st.session_state.active_tab in tab_keys:
        default_idx = tab_keys.index(st.session_state.active_tab)

    selected_label = st.selectbox(
        "Navigasi",
        options=tab_labels,
        index=default_idx,
        label_visibility="collapsed"
    )

    selected_tab = label_to_key[selected_label]
    st.session_state.active_tab = selected_tab

    # =========================
    # WORKSPACE
    # =========================
    st.markdown('<div class="sb-section">Workspace</div>', unsafe_allow_html=True)

    st.toggle("Tampilkan panel insight", key="panel_open")
    st.toggle("Mode compare 2 provinsi", key="compare_mode")

    # =========================
    # QUICK ACTIONS
    # =========================
    st.markdown('<div class="sb-section">Quick Actions</div>', unsafe_allow_html=True)

    bm_name = st.text_input(
        "Nama bookmark",
        placeholder="e.g. Peta 2023 PDRB",
        key="bm_input"
    )

    if st.button("💾 Simpan Bookmark", use_container_width=True):
        if bm_name:
            item = {"name": bm_name, "tab": st.session_state.active_tab}
            st.session_state.bookmarks = [
                b for b in st.session_state.bookmarks
                if b.get("name") != bm_name
            ]
            st.session_state.bookmarks.insert(0, item)
            st.session_state.bookmarks = st.session_state.bookmarks[:8]
            st.success("Tersimpan!")

    if st.button("🗑 Hapus semua bookmark", use_container_width=True):
        st.session_state.bookmarks = []
        st.rerun()

    # 🔥 RESET CACHE (IMPORTANT)
    if st.button("🔄 Reset Cache Data", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache dibersihkan!")
        st.rerun()

    # =========================
    # BOOKMARKS
    # =========================
    st.markdown('<div class="sb-section">Bookmarks</div>', unsafe_allow_html=True)
    render_bookmarks()

    # =========================
    # INFO
    # =========================
    st.markdown('<div class="sb-section">Info</div>', unsafe_allow_html=True)

    st.caption("Sumber: BPS Indonesia")

    st.markdown(
        """<div style='font-size:11px;color:#6b7a99;line-height:1.8;margin-top:4px;'>
    ✅ Tooltip interaktif setiap chart<br>
    ✅ Filter per tab<br>
    ✅ Insight otomatis<br>
    ✅ AI Analytics (forecast, cluster)<br>
    ✅ Bookmark & Compare mode<br>
    ✅ Export CSV
    </div>""",
        unsafe_allow_html=True,
    )

# =========================
# PAGE TITLE (ACTIVE TAB)
# =========================
st.markdown(
    f"<div style='margin: 2px 0 16px; font-size: 14px; color: #9aa5be;'><b>{TAB_LABELS[st.session_state.active_tab]}</b></div>",
    unsafe_allow_html=True
)

# =========================================================
# TAB: RINGKASAN
# =========================================================
def render_summary():
    years = years_of(pdrb)
    with st.container():
        st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
        fc1, fc2 = st.columns(2)
        with fc1:
            start_year = st.selectbox("📅 Tahun Awal", years, index=0, key="sum_s")
        with fc2:
            end_year = st.selectbox("📅 Tahun Akhir", years, index=len(years) - 1, key="sum_e")
        st.markdown('</div>', unsafe_allow_html=True)

    def yf(df):
        return yr_filter(df, start_year, end_year)

    pdrb_ly = max(years_of(pdrb))
    tpt_ly = max(years_of(tpt))

    def nat(df):
        return df[df["Provinsi"].str.upper() == "INDONESIA"]

    def safe_val(df, col, yr):
        r = df[df["Tahun"] == yr][col].dropna()
        return float(r.iloc[0]) if not r.empty else None

    def delta_kpi(df, col, yr):
        a = safe_val(df, col, yr)
        b = safe_val(df, col, yr - 1)
        return (a - b) if a is not None and b is not None else None

    # PDRB
    v_pdrb = safe_val(nat(pdrb), "PDRB_PerKapita_RibuRupiah", pdrb_ly)
    d_pdrb = None
    if v_pdrb is not None:
        p = safe_val(nat(pdrb), "PDRB_PerKapita_RibuRupiah", pdrb_ly - 1)
        d_pdrb = ((v_pdrb - p) / p * 100) if p else None

    # TPT
    tpt_nat = nat(tpt)[nat(tpt)["Periode"] == "Agustus"]
    v_tpt = safe_val(tpt_nat, "TPT_Persen", tpt_ly)
    d_tpt = delta_kpi(tpt_nat, "TPT_Persen", tpt_ly)          # in percentage points

    # Kemiskinan
    msk_nat = nat(miskin)[
        (nat(miskin)["Daerah"] == "Jumlah") &
        (nat(miskin)["Semester"] == "Semester 1 (Maret)")
    ]
    msk_sorted = msk_nat.sort_values("Tahun")
    msk_vals = msk_sorted["Persen_Penduduk_Miskin"].dropna()
    v_msk = float(msk_vals.iloc[-1]) if not msk_vals.empty else None
    p_msk = float(msk_vals.iloc[-2]) if len(msk_vals) >= 2 else None
    d_msk = (v_msk - p_msk) if v_msk is not None and p_msk is not None else None

    # Gini
    gini_nat = nat(gini)[
        (nat(gini)["Daerah"] == "Perkotaan+Perdesaan") &
        (nat(gini)["Semester"] == "Semester 1 (Maret)")
    ].sort_values("Tahun")
    gini_vals = gini_nat["Gini_Ratio"].dropna()
    v_gini = float(gini_vals.iloc[-1]) if not gini_vals.empty else None
    p_gini = float(gini_vals.iloc[-2]) if len(gini_vals) >= 2 else None
    d_gini = (v_gini - p_gini) if v_gini is not None and p_gini is not None else None

    # Inflasi
    inf_nat = nat(inflasi).groupby("Tahun", as_index=False)["Inflasi_YoY_Persen"].mean().sort_values("Tahun")
    inf_vals = inf_nat["Inflasi_YoY_Persen"].dropna()
    v_inf = float(inf_vals.iloc[-1]) if not inf_vals.empty else None
    p_inf = float(inf_vals.iloc[-2]) if len(inf_vals) >= 2 else None
    d_inf = (v_inf - p_inf) if v_inf is not None and p_inf is not None else None

    # KPI cards – now all use reverse=False (positive = green ▲, negative = red ▼)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi("PDRB per Kapita", f"Rp {v_pdrb:,.0f} Ribu" if v_pdrb else "—",
                       f"{pdrb_ly}", d_pdrb, "%"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Pengangguran TPT", f"{v_tpt:.2f}%" if v_tpt else "—",
                       f"Agustus {tpt_ly}", d_tpt, "%"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Kemiskinan", f"{v_msk:.2f}%" if v_msk else "—",
                       "Maret · Nasional", d_msk, "%"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Gini Ratio", f"{v_gini:.3f}" if v_gini else "—",
                       "Kota+Desa", d_gini, "", delta_fmt=".3f"), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi("Inflasi YoY", f"{v_inf:.2f}%" if v_inf else "—",
                       f"Rata-rata {inf_nat['Tahun'].max() if not inf_nat.empty else '—'}", d_inf, "%"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: PDRB trend and TPT + poverty
    col_l, col_r = st.columns(2)
    with col_l:
        sec("Tren PDRB per Kapita Nasional", f"{start_year}–{end_year}")
        pdrb_line = yf(nat(pdrb)).sort_values("Tahun")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pdrb_line["Tahun"], y=pdrb_line["PDRB_PerKapita_RibuRupiah"],
            fill="tozeroy", mode="lines+markers",
            line=dict(color=COLORS["green"], width=2.8),
            fillcolor="rgba(34,211,164,0.08)",
            marker=dict(size=6, color=COLORS["green"]),
            hovertemplate="<b>Tahun %{x}</b><br>PDRB/Kapita: <b>Rp %{y:,.0f} Ribu</b><extra></extra>"
        ))
        apply_layout(fig, h=270, yaxis=dict(title="Rp Ribu", gridcolor="rgba(255,255,255,0.04)"))
        st.plotly_chart(fig, use_container_width=True)
        if len(pdrb_line) >= 2:
            fv = float(pdrb_line["PDRB_PerKapita_RibuRupiah"].iloc[0])
            lv = float(pdrb_line["PDRB_PerKapita_RibuRupiah"].iloc[-1])
            n = len(pdrb_line) - 1 or 1
            insight_callout("Insight PDRB", [
                f"Pertumbuhan kumulatif: <b>{(lv - fv) / fv * 100:.1f}%</b> ({start_year}→{end_year})",
                f"CAGR ≈ <b>{((lv / fv) ** (1 / n) - 1) * 100:.1f}%</b>/tahun",
                f"Nilai terakhir: <b>Rp {lv:,.0f} Ribu</b>",
            ], tone="good")

    with col_r:
        sec("TPT & Kemiskinan Nasional", "Indonesia")
        tpt_f = yf(nat(tpt)[nat(tpt)["Periode"] == "Agustus"]).sort_values("Tahun")
        msk_f = yf(nat(miskin)[
            (nat(miskin)["Daerah"] == "Jumlah") &
            (nat(miskin)["Semester"] == "Semester 1 (Maret)")
        ]).sort_values("Tahun")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=tpt_f["Tahun"], y=tpt_f["TPT_Persen"], name="TPT (%)",
            mode="lines+markers", line=dict(color=COLORS["red"], width=2.8), marker=dict(size=6),
            hovertemplate="<b>TPT %{x}</b><br>Pengangguran: <b>%{y:.2f}%</b><extra></extra>"
        ))
        fig2.add_trace(go.Scatter(
            x=msk_f["Tahun"], y=msk_f["Persen_Penduduk_Miskin"], name="Kemiskinan (%)",
            mode="lines+markers", line=dict(color=COLORS["purple"], width=2.8, dash="dot"), marker=dict(size=6),
            hovertemplate="<b>Kemiskinan %{x}</b><br>Penduduk miskin: <b>%{y:.2f}%</b><extra></extra>"
        ))
        apply_layout(fig2, h=270, legend_h=True, yaxis=dict(title="%", gridcolor="rgba(255,255,255,0.04)"))
        st.plotly_chart(fig2, use_container_width=True)
        if not tpt_f.empty and not msk_f.empty:
            tpt_trend = "turun ✅" if float(tpt_f["TPT_Persen"].iloc[-1]) < float(tpt_f["TPT_Persen"].iloc[0]) else "naik ⚠️"
            msk_trend = "turun ✅" if float(msk_f["Persen_Penduduk_Miskin"].iloc[-1]) < float(msk_f["Persen_Penduduk_Miskin"].iloc[0]) else "naik ⚠️"
            insight_callout("Insight TPT & Kemiskinan", [
                f"TPT sepanjang periode: <b>{tpt_trend}</b>",
                f"Kemiskinan sepanjang periode: <b>{msk_trend}</b>",
                f"TPT terkini: <b>{float(tpt_f['TPT_Persen'].iloc[-1]):.2f}%</b>",
            ], tone="info")

    # Row 2: Top 10 PDRB and dual-axis Gini/Inflasi
    col_l2, col_r2 = st.columns(2)
    with col_l2:
        sec("Top 10 Provinsi PDRB/Kapita", f"Tahun {pdrb_ly}")
        top10 = non_country(pdrb)[pdrb["Tahun"] == pdrb_ly].nlargest(10, "PDRB_PerKapita_RibuRupiah").copy()
        top10["label"] = top10["Provinsi"].map(clean_label)
        fig3 = go.Figure(go.Bar(
            x=top10["PDRB_PerKapita_RibuRupiah"], y=top10["label"], orientation="h",
            marker=dict(
                color=top10["PDRB_PerKapita_RibuRupiah"],
                colorscale=[[0, "#0f6e56"], [0.5, "#1D9E75"], [1, "#22d3a4"]],
                showscale=False
            ),
            hovertemplate="<b>%{y}</b><br>PDRB/Kapita: <b>Rp %{x:,.0f} Ribu</b><br>Rank: Top 10<extra></extra>",
            text=top10["PDRB_PerKapita_RibuRupiah"].apply(lambda v: f"Rp {v:,.0f}"),
            textposition="outside", textfont=dict(size=9, color="#9aa5be")
        ))
        fig3.update_yaxes(categoryorder="total ascending")
        apply_layout(fig3, h=320, xaxis=dict(title="Rp Ribu", gridcolor="rgba(255,255,255,0.04)"), yaxis=dict(title=""))
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        sec("Gini Ratio & Inflasi Nasional", "Indonesia · dual axis")
        gini_f = yf(nat(gini)[
            (nat(gini)["Daerah"] == "Perkotaan+Perdesaan") &
            (nat(gini)["Semester"] == "Semester 1 (Maret)")
        ]).sort_values("Tahun")
        inf_f = yf(nat(inflasi).groupby("Tahun", as_index=False)["Inflasi_YoY_Persen"].mean())
        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        fig4.add_trace(go.Scatter(
            x=gini_f["Tahun"], y=gini_f["Gini_Ratio"], name="Gini",
            mode="lines+markers", line=dict(color=COLORS["warn"], width=2.8), marker=dict(size=6),
            hovertemplate="<b>Gini %{x}</b><br>Indeks: <b>%{y:.3f}</b><extra></extra>"
        ), secondary_y=False)
        fig4.add_trace(go.Scatter(
            x=inf_f["Tahun"], y=inf_f["Inflasi_YoY_Persen"], name="Inflasi (%)",
            mode="lines+markers", line=dict(color=COLORS["orange"], width=2.8, dash="dash"), marker=dict(size=6),
            hovertemplate="<b>Inflasi %{x}</b><br>YoY: <b>%{y:.2f}%</b><extra></extra>"
        ), secondary_y=True)
        fig4.update_yaxes(title_text="Gini Ratio", secondary_y=False, gridcolor="rgba(255,255,255,0.04)", color="#9aa5be")
        fig4.update_yaxes(title_text="Inflasi YoY (%)", secondary_y=True, color="#9aa5be")
        apply_layout(fig4, h=320, legend_h=True)
        st.plotly_chart(fig4, use_container_width=True)

    # Insight panel (if enabled)
    if st.session_state.panel_open:
        bullets = []
        if d_pdrb is not None:
            bullets.append(f"PDRB/kapita {'naik' if d_pdrb > 0 else 'turun'} <b>{abs(d_pdrb):.2f}%</b> vs tahun sebelumnya")
        if v_tpt is not None:
            bullets.append(f"TPT nasional: <b>{v_tpt:.2f}%</b> pada Agustus {tpt_ly}")
        if v_msk is not None:
            bullets.append(f"Kemiskinan nasional: <b>{v_msk:.2f}%</b> (Maret terbaru)")
        if v_gini is not None:
            trend = "meningkat ⚠️" if d_gini and d_gini > 0 else "membaik ✅" if d_gini is not None else "—"
            bullets.append(f"Gini ratio: <b>{v_gini:.3f}</b> — ketimpangan {trend}")
        if v_inf is not None:
            bullets.append(f"Inflasi rata-rata: <b>{v_inf:.2f}%</b> {'— di atas target BI ⚠️' if v_inf > 4 else '— dalam rentang wajar ✅'}")
        insight_callout("Ringkasan Cepat — Nasional", bullets, tone="info")

# =========================================================
# TAB: PETA INTERAKTIF
# =========================================================
def render_map():
    years = years_of(pdrb)
    with st.container():
        st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
        fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 2])
        with fc1:
            map_ind = st.selectbox(
                "📊 Indikator Peta",
                ["PDRB/Kapita", "Pengangguran (TPT)", "Kemiskinan", "Gini Ratio", "Inflasi"],
                key="map_ind"
            )
        with fc2:
            map_year = st.selectbox("📅 Tahun", years[::-1], key="map_year")
        with fc3:
            map_style = st.selectbox(
                "🎨 Skema warna",
                ["OrRd", "Teal", "Viridis", "Plasma", "Cividis", "Reds", "Purp", "YlOrBr"],
                key="map_style"
            )
        with fc4:
            mapbox_style = st.selectbox("🗺️ Map Style", ["carto-darkmatter", "carto-positron", "open-street-map"])
        st.markdown('</div>', unsafe_allow_html=True)

    use_34 = map_year <= 2022
    geojson_active = geojson_34 if use_34 else geojson_38
    geo_label = "34 Provinsi (pra-pemekaran)" if use_34 else "38 Provinsi"
    st.markdown(f"<div class='chip'>📌 {geo_label}</div>", unsafe_allow_html=True)

    df_ind, title, cs_hint, unit = map_indicator(map_ind)
    is_pdrb = map_ind == "PDRB/Kapita"
    cs = map_style if map_style not in ["Teal", "Purp"] else cs_hint

    df_ind = df_ind.copy()
    df_ind["Provinsi"] = df_ind["Provinsi"].astype(str).str.strip().str.upper()
    df_ind["provinsi_name"] = df_ind["Provinsi"].map(NAME_MAP)
    df_map = df_ind[df_ind["Tahun"] == map_year].copy()

    if use_34:
        df_map = df_map[~df_map["Provinsi"].isin(PROV_BARU)]
    df_map = df_map.dropna(subset=["provinsi_name", "value"])

    if df_map.empty:
        st.warning("Tidak ada data untuk tahun dan indikator ini.")
        return
    if geojson_active is None:
        st.error("File GeoJSON tidak ditemukan.")
        return

    df_map["rank"] = df_map["value"].rank(ascending=False, method="dense").astype(int)
    df_map["share"] = (df_map["value"] / df_map["value"].sum() * 100).round(2)
    df_map["delta_vs_mean"] = (df_map["value"] - df_map["value"].mean()).round(2)

    if is_pdrb:
        df_map["value_display"] = df_map["value"].apply(format_rupiah_auto)
        df_map["delta_display"] = df_map["delta_vs_mean"].apply(
            lambda x: format_delta_display(x, unit="Rp")
        )
    else:
        df_map["value_display"] = df_map["value"].apply(lambda x: fmt_v(x, 2, unit))
        df_map["delta_display"] = df_map["delta_vs_mean"].apply(
            lambda x: format_delta_display(x, unit=unit, digits=2)
        )

    sec(f"🗺️ Peta Choropleth — {map_ind} · {map_year}", geo_label)

    fig_map = px.choropleth_mapbox(
        df_map,
        geojson=geojson_active,
        locations="provinsi_name",
        featureidkey="properties.PROVINSI",
        color="value",
        color_continuous_scale=cs,
        range_color=(float(df_map["value"].min()), float(df_map["value"].max())),
        mapbox_style=mapbox_style,
        zoom=3.6,
        center={"lat": -2.5, "lon": 118},
        opacity=0.88,
        labels={"value": title},
        custom_data=["rank", "share", "delta_display", "provinsi_name", "value_display"]
    )

    if is_pdrb:
        hovertemplate = (
            "<b>%{{customdata[3]}}</b><br>"
            f"{title}: <b>%{{customdata[4]}}</b><br>"
            "Rank Nasional: <b>#%{{customdata[0]}}</b> dari " + str(len(df_map)) + "<br>"
            "Share: <b>%{{customdata[1]:.2f}}%</b><br>"
            "vs rata-rata: <b>%{{customdata[2]}}</b><extra></extra>"
        )
    else:
        hovertemplate = (
            "<b>%{{customdata[3]}}</b><br>"
            f"{title}: <b>%{{z:,.2f}} {unit}</b><br>"
            "Rank Nasional: <b>#%{{customdata[0]}}</b> dari " + str(len(df_map)) + "<br>"
            "Share: <b>%{{customdata[1]:.2f}}%</b><br>"
            "vs rata-rata: <b>%{{customdata[2]}}</b><extra></extra>"
        )

    fig_map.update_traces(hovertemplate=hovertemplate)

    fig_map.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=540,
        paper_bgcolor="rgba(11,15,25,0)",
        coloraxis_colorbar=dict(
            thickness=12,
            len=0.62,
            title=dict(text=("Rp" if is_pdrb else unit), font=dict(color="#9aa5be")),
            tickfont=dict(color="#9aa5be"),
            bgcolor="rgba(17,24,39,0.82)",
            bordercolor="rgba(255,255,255,.1)",
            borderwidth=1,
        )
    )

    st.plotly_chart(fig_map, use_container_width=True)

    avg_v = df_map["value"].mean()
    max_v = df_map["value"].max()
    min_v = df_map["value"].min()
    std_v = df_map["value"].std()
    top_p = df_map.loc[df_map["value"].idxmax(), "provinsi_name"]
    bot_p = df_map.loc[df_map["value"].idxmin(), "provinsi_name"]

    format_main = format_rupiah_auto if is_pdrb else (lambda x: fmt_v(x, 2, unit))

    insight_callout(
        f"Insight Peta — {map_ind} {map_year}",
        [
            f"<b>Tertinggi:</b> {top_p} ({format_main(max_v)})",
            f"<b>Terendah:</b> {bot_p} ({format_main(min_v)})",
            f"<b>Rata-rata:</b> {format_main(avg_v)} · <b>Std dev:</b> {format_main(std_v)}",
            f"<b>CV:</b> {std_v / avg_v:.2f} — ketimpangan antarprovinsi {'tinggi ⚠️' if std_v / avg_v > 0.4 else 'sedang 🟡' if std_v / avg_v > 0.2 else 'rendah ✅'}",
        ],
        tone="info"
    )

    col_rank, col_stat = st.columns([2.2, 1])
    with col_rank:
        sec("Ranking Provinsi", f"{len(df_map)} provinsi")
        df_rank = df_map[["rank", "provinsi_name", "value", "share"]].sort_values("rank").copy()
        df_rank.columns = ["Rank", "Provinsi", title, "Share (%)"]

        if is_pdrb:
            df_rank[title] = df_rank[title].apply(format_rupiah_auto)
            st.dataframe(df_rank, use_container_width=True, height=320)
        else:
            st.dataframe(
                df_rank.style.background_gradient(cmap="coolwarm", subset=[title]).format({title: "{:.2f}", "Share (%)": "{:.2f}"}),
                use_container_width=True,
                height=320,
            )

    with col_stat:
        sec("Quick Stats", str(map_year))
        st.metric("Rata-rata", format_main(avg_v))
        st.metric("Tertinggi", format_main(max_v), top_p)
        st.metric("Terendah", format_main(min_v), bot_p)
        st.metric("Std Dev", format_main(std_v))
        above = len(df_map[df_map["value"] > avg_v])
        st.metric("Di atas rata-rata", f"{above} provinsi")

# =========================================================
# TAB: TREN WAKTU
# =========================================================
def render_trend():
    provinces = get_prov_list(pdrb)
    defaults_prov = [p for p in ["DKI JAKARTA", "JAWA BARAT", "JAWA TIMUR", "SUMATERA UTARA", "KALIMANTAN TIMUR"] if p in provinces][:5]

    with st.container():
        st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
        fc1, fc2, fc3, fc4 = st.columns([3, 1, 1, 1])
        with fc1:
            prov_sel = st.multiselect("🏙️ Provinsi (maks 10)", provinces, default=defaults_prov, key="trend_prov", max_selections=10)
        with fc2:
            metric_sel = st.selectbox("Indikator", ["PDRB/Kapita", "TPT", "Kemiskinan", "Gini Ratio", "Inflasi"], key="trend_metric")
        with fc3:
            years = years_of(pdrb)
            start_year = st.selectbox("Dari", years, index=0, key="trend_s")
        with fc4:
            end_year = st.selectbox("Sampai", years, index=len(years) - 1, key="trend_e")
        st.markdown('</div>', unsafe_allow_html=True)

    if not prov_sel:
        st.info("👈 Pilih minimal 1 provinsi.")
        return

    def yf(df):
        return yr_filter(df, start_year, end_year)

    if metric_sel == "PDRB/Kapita":
        df = yf(non_country(pdrb)[non_country(pdrb)["Provinsi"].isin(prov_sel)]).copy()
        df["label"] = df["Provinsi"].map(clean_label)
        sec("PDRB per Kapita", f"{start_year}–{end_year}")
        fig = px.line(df, x="Tahun", y="PDRB_PerKapita_RibuRupiah", color="label", markers=True, color_discrete_sequence=PROV_PALETTE, labels={"label": ""})
        fig.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Tahun %{x}<br>PDRB/Kapita: <b>Rp %{y:,.0f} Ribu</b><extra></extra>", line_width=2.3, marker_size=5)
        apply_layout(fig, h=320, legend_h=True, yaxis=dict(title="Rp Ribu", gridcolor="rgba(255,255,255,0.04)"))
        st.plotly_chart(fig, use_container_width=True)

    elif metric_sel == "TPT":
        df = yf(non_country(tpt)[(non_country(tpt)["Periode"] == "Agustus") & (non_country(tpt)["Provinsi"].isin(prov_sel))]).copy()
        df["label"] = df["Provinsi"].map(clean_label)
        sec("Tingkat Pengangguran Terbuka (TPT)", f"Agustus · {start_year}–{end_year}", COLORS["red"])
        fig = px.line(df, x="Tahun", y="TPT_Persen", color="label", markers=True, color_discrete_sequence=PROV_PALETTE, labels={"label": ""})
        fig.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Tahun %{x}<br>TPT: <b>%{y:.2f}%</b><extra></extra>", line_width=2.3, marker_size=5)
        apply_layout(fig, h=320, legend_h=True, yaxis=dict(title="%", gridcolor="rgba(255,255,255,0.04)"))
        st.plotly_chart(fig, use_container_width=True)

    elif metric_sel == "Kemiskinan":
        df = yf(non_country(miskin)[(non_country(miskin)["Daerah"] == "Jumlah") & (non_country(miskin)["Semester"] == "Semester 1 (Maret)") & (non_country(miskin)["Provinsi"].isin(prov_sel))]).copy()
        df["label"] = df["Provinsi"].map(clean_label)
        sec("Kemiskinan (%)", f"Maret · {start_year}–{end_year}", COLORS["purple"])
        fig = px.line(df, x="Tahun", y="Persen_Penduduk_Miskin", color="label", markers=True, color_discrete_sequence=PROV_PALETTE, labels={"label": ""})
        fig.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Tahun %{x}<br>Kemiskinan: <b>%{y:.2f}%</b><extra></extra>", line_width=2.3, marker_size=5)
        apply_layout(fig, h=320, legend_h=True, yaxis=dict(title="%", gridcolor="rgba(255,255,255,0.04)"))
        st.plotly_chart(fig, use_container_width=True)

    elif metric_sel == "Gini Ratio":
        df = yf(non_country(gini)[(non_country(gini)["Daerah"] == "Perkotaan+Perdesaan") & (non_country(gini)["Semester"] == "Semester 1 (Maret)") & (non_country(gini)["Provinsi"].isin(prov_sel))]).copy()
        df["label"] = df["Provinsi"].map(clean_label)
        sec("Gini Ratio", f"Maret · {start_year}–{end_year}", COLORS["warn"])
        fig = px.line(df, x="Tahun", y="Gini_Ratio", color="label", markers=True, color_discrete_sequence=PROV_PALETTE, labels={"label": ""})
        fig.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Tahun %{x}<br>Gini: <b>%{y:.3f}</b><extra></extra>", line_width=2.3, marker_size=5)
        apply_layout(fig, h=320, legend_h=True, yaxis=dict(title="Gini", gridcolor="rgba(255,255,255,0.04)"))
        st.plotly_chart(fig, use_container_width=True)

    else:
        df = yf(non_country(inflasi)[non_country(inflasi)["Provinsi"].isin(prov_sel)].groupby(["Provinsi", "Tahun"], as_index=False)["Inflasi_YoY_Persen"].mean()).copy()
        df["label"] = df["Provinsi"].map(clean_label)
        sec("Inflasi YoY (%)", f"{start_year}–{end_year}", COLORS["orange"])
        fig = px.line(df, x="Tahun", y="Inflasi_YoY_Persen", color="label", markers=True, color_discrete_sequence=PROV_PALETTE, labels={"label": ""})
        fig.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Tahun %{x}<br>Inflasi: <b>%{y:.2f}%</b><extra></extra>", line_width=2.3, marker_size=5)
        fig.add_hline(y=3.0, line_dash="dot", line_color="rgba(251,191,36,0.5)", annotation_text="Target BI 3%", annotation_font=dict(color="#fbbf24", size=10))
        apply_layout(fig, h=320, legend_h=True, yaxis=dict(title="%", gridcolor="rgba(255,255,255,0.04)"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sec("Heatmap Inflasi Bulanan", "Setiap sel = rata-rata YoY per bulan")
    month_order = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
    inf_m = yf(non_country(inflasi)[non_country(inflasi)["Provinsi"].isin(prov_sel)]).copy()
    inf_m["label"] = inf_m["Provinsi"].map(clean_label)
    if not inf_m.empty:
        inf_m["m_idx"] = inf_m["Bulan"].map({m: i for i, m in enumerate(month_order)})
        inf_m = inf_m.sort_values(["Tahun", "m_idx"])
        inf_m["period"] = inf_m["Tahun"].astype(str) + "-" + inf_m["Bulan"].str[:3]
        heat_piv = inf_m.pivot_table(index="label", columns="period", values="Inflasi_YoY_Persen", aggfunc="mean")
        fig_heat = px.imshow(heat_piv, color_continuous_scale="RdYlGn_r", aspect="auto", labels={"color": "Inflasi (%)", "x": "Periode", "y": "Provinsi"})
        fig_heat.update_traces(hovertemplate="<b>%{y}</b><br>%{x}<br>Inflasi: <b>%{z:.2f}%</b><extra></extra>")
        fig_heat.update_layout(height=max(250, len(prov_sel) * 35 + 60), **{k: v for k, v in PLOT_LAYOUT.items() if k not in ["xaxis", "yaxis", "margin"]}, margin=dict(t=10, b=40, l=10, r=10), xaxis=dict(tickangle=-45, tickfont=dict(size=8)), coloraxis_colorbar=dict(thickness=10, len=0.6, title=dict(text="%", font=dict(color="#9aa5be")), tickfont=dict(color="#9aa5be")))
        st.plotly_chart(fig_heat, use_container_width=True)

# =========================================================
# TAB: PERBANDINGAN
# =========================================================
def render_comparison():
    with st.container():
        st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns([3, 1, 2])
        with fc1:
            ind_cmp = st.selectbox("📊 Indikator", ["PDRB per Kapita (Rp Ribu)", "Pengangguran TPT (%)", "Kemiskinan (%)", "Gini Ratio", "Inflasi YoY (%)"], key="cmp_ind")
        with fc2:
            yr_cmp = st.selectbox("📅 Tahun", years_of(pdrb)[::-1], key="cmp_yr")
        with fc3:
            sort_asc = st.radio("Urutan", ["Descending ↓", "Ascending ↑"], horizontal=True, key="cmp_sort")
        st.markdown('</div>', unsafe_allow_html=True)

    # fixed comparison data
    if ind_cmp == "PDRB per Kapita (Rp Ribu)":
        df_c = non_country(pdrb)[pdrb["Tahun"] == yr_cmp][["Provinsi", "PDRB_PerKapita_RibuRupiah"]].copy()
        df_c.columns = ["Provinsi", "Nilai"]
        unit_c = "dalam Ribuan"
        color_c = COLORS["green"]
        invert_better = False
    elif ind_cmp == "Pengangguran TPT (%)":
        df_c = non_country(tpt)[(non_country(tpt)["Periode"] == "Agustus") & (non_country(tpt)["Tahun"] == yr_cmp)][["Provinsi", "TPT_Persen"]].copy()
        df_c.columns = ["Provinsi", "Nilai"]
        unit_c = "%"
        color_c = COLORS["red"]
        invert_better = True
    elif ind_cmp == "Kemiskinan (%)":
        df_c = non_country(miskin)[(non_country(miskin)["Daerah"] == "Jumlah") & (non_country(miskin)["Semester"] == "Semester 1 (Maret)") & (non_country(miskin)["Tahun"] == yr_cmp)][["Provinsi", "Persen_Penduduk_Miskin"]].copy()
        df_c.columns = ["Provinsi", "Nilai"]
        unit_c = "%"
        color_c = COLORS["purple"]
        invert_better = True
    elif ind_cmp == "Gini Ratio":
        df_c = non_country(gini)[(non_country(gini)["Daerah"] == "Perkotaan+Perdesaan") & (non_country(gini)["Semester"] == "Semester 1 (Maret)") & (non_country(gini)["Tahun"] == yr_cmp)][["Provinsi", "Gini_Ratio"]].copy()
        df_c.columns = ["Provinsi", "Nilai"]
        unit_c = ""
        color_c = COLORS["warn"]
        invert_better = True
    else:
        df_c = non_country(inflasi)[non_country(inflasi)["Tahun"] == yr_cmp].groupby("Provinsi", as_index=False)["Inflasi_YoY_Persen"].mean()
        df_c.columns = ["Provinsi", "Nilai"]
        unit_c = "%"
        color_c = COLORS["orange"]
        invert_better = True

    df_c = df_c.dropna(subset=["Nilai"]).copy()
    if df_c.empty:
        st.warning("Data tidak tersedia.")
        return

    df_c = df_c.sort_values("Nilai", ascending=sort_asc.startswith("Asc")).copy()
    df_c["label"] = df_c["Provinsi"].map(clean_label)
    avg_cmp = df_c["Nilai"].mean()
    df_c["delta_vs_avg"] = df_c["Nilai"] - avg_cmp
    df_c["status"] = np.where(df_c["Nilai"] >= avg_cmp, "Di atas rata-rata", "Di bawah rata-rata")
    df_c["score_100"] = normalize_series(df_c["Nilai"], invert=invert_better)

    sec(f"Perbandingan Provinsi — {ind_cmp}", f"Tahun {yr_cmp}", color_c)

    df_c["bar_color"] = np.where(df_c["Nilai"] >= avg_cmp, color_c, "rgba(255,255,255,0.16)")
    fig_c = go.Figure(go.Bar(
        x=df_c["label"], y=df_c["Nilai"],
        marker_color=df_c["bar_color"],
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"{ind_cmp}: <b>%{{y:.2f}}{unit_c}</b><br>"
            "vs rata-rata: <b>%{customdata[0]:+.2f}</b><br>"
            "Status: <b>%{customdata[1]}</b><extra></extra>"
        ),
        customdata=np.stack([df_c["delta_vs_avg"], df_c["status"]], axis=-1),
        text=df_c["Nilai"].round(2), textposition="outside",
        textfont=dict(size=8, color="#6b7a99")
    ))
    fig_c.add_hline(y=avg_cmp, line_dash="dot", line_color="rgba(255,255,255,0.35)", annotation_text=f"  Rata-rata: {avg_cmp:.2f}{unit_c}", annotation_font=dict(color="#9aa5be", size=11), annotation_position="top left")
    apply_layout(fig_c, h=430, xaxis=dict(tickangle=-45, title="", gridcolor="rgba(255,255,255,0.04)"), yaxis=dict(title=ind_cmp, gridcolor="rgba(255,255,255,0.04)"))
    st.plotly_chart(fig_c, use_container_width=True)

    st.caption("Catatan: untuk indikator dengan skala sangat berbeda, gunakan radar/normalized score agar perbandingan lebih fair.")

    # normalized bar chart as alternative
    norm_df = pd.DataFrame({"Provinsi": df_c["label"].tolist() * 2,
                            "Indikator": [ind_cmp] * len(df_c) * 2,
                            "Skor": list(df_c["score_100"]) + list(df_c["score_100"])})
    norm_df["Kelompok"] = [clean_label(st.session_state.get("cmp_a", "Provinsi A"))] * len(df_c) + [clean_label(st.session_state.get("cmp_b", "Provinsi B"))] * len(df_c)

    above_n = len(df_c[df_c["Nilai"] >= avg_cmp])
    top3_names = ", ".join(df_c.nlargest(3, "Nilai")["label"].tolist())
    insight_callout(f"Insight — {ind_cmp} {yr_cmp}", [
        f"<b>{above_n}</b> dari {len(df_c)} provinsi di atas rata-rata ({avg_cmp:.2f}{unit_c})",
        f"Gap maks–min: <b>{df_c['Nilai'].max() - df_c['Nilai'].min():.2f}{unit_c}</b>",
        f"Nilai tertinggi: <b>{top3_names}</b>",
    ], tone="info")

    # Compare mode — fair duel 2 provinsi
    if st.session_state.compare_mode:
        st.markdown("<br>", unsafe_allow_html=True)
        sec("Mode Compare — Duel 2 Provinsi", "radar + raw value")
        prov_list = get_prov_list(pdrb)
        ca, cb = st.columns(2)
        with ca:
            prov_a = st.selectbox("Provinsi A", prov_list, index=0, key="cmp_a")
        with cb:
            prov_b = st.selectbox("Provinsi B", prov_list, index=min(1, len(prov_list) - 1), key="cmp_b")

        inflasi_agg = inflasi.groupby(["Provinsi", "Tahun"], as_index=False)["Inflasi_YoY_Persen"].mean()

        metric_defs = [
            ("Rp", "PDRB/Kapita", non_country(pdrb)[pdrb["Tahun"] == yr_cmp][["Provinsi", "PDRB_PerKapita_RibuRupiah"]].rename(columns={"PDRB_PerKapita_RibuRupiah": "Nilai"}), False),
            ("TPT (%)", non_country(tpt)[(non_country(tpt)["Periode"] == "Agustus") & (non_country(tpt)["Tahun"] == yr_cmp)][["Provinsi", "TPT_Persen"]].rename(columns={"TPT_Persen": "Nilai"}), True, "%"),
            ("Kemiskinan (%)", non_country(miskin)[(non_country(miskin)["Daerah"] == "Jumlah") & (non_country(miskin)["Semester"] == "Semester 1 (Maret)") & (non_country(miskin)["Tahun"] == yr_cmp)][["Provinsi", "Persen_Penduduk_Miskin"]].rename(columns={"Persen_Penduduk_Miskin": "Nilai"}), True, "%"),
            ("Gini", non_country(gini)[(non_country(gini)["Daerah"] == "Perkotaan+Perdesaan") & (non_country(gini)["Semester"] == "Semester 1 (Maret)") & (non_country(gini)["Tahun"] == yr_cmp)][["Provinsi", "Gini_Ratio"]].rename(columns={"Gini_Ratio": "Nilai"}), True, ""),
            ("Inflasi (%)", inflasi_agg[inflasi_agg["Tahun"] == yr_cmp][["Provinsi", "Inflasi_YoY_Persen"]].rename(columns={"Inflasi_YoY_Persen": "Nilai"}), True, "%"),
        ]

        radar_rows = []
        raw_rows = []
        for metric_name, dfm, invert, unit in metric_defs:
            dfm = dfm.dropna(subset=["Nilai"]).copy()
            if dfm.empty:
                continue
            dfm["norm"] = normalize_series(dfm["Nilai"], invert=invert)
            va = dfm.loc[dfm["Provinsi"] == prov_a, "Nilai"]
            vb = dfm.loc[dfm["Provinsi"] == prov_b, "Nilai"]
            na = dfm.loc[dfm["Provinsi"] == prov_a, "norm"]
            nb = dfm.loc[dfm["Provinsi"] == prov_b, "norm"]
            if not va.empty and not vb.empty and not na.empty and not nb.empty:
                raw_rows.append({
                    "Indikator": metric_name,
                    clean_label(prov_a): float(va.iloc[0]),
                    clean_label(prov_b): float(vb.iloc[0]),
                    "Unit": unit,
                })
                radar_rows.append({
                    "Indikator": metric_name,
                    clean_label(prov_a): float(na.iloc[0]),
                    clean_label(prov_b): float(nb.iloc[0]),
                })

        radar_df = pd.DataFrame(radar_rows)
        raw_df = pd.DataFrame(raw_rows)

        if not radar_df.empty:
            categories = radar_df["Indikator"].tolist()
            a_vals = radar_df[clean_label(prov_a)].tolist()
            b_vals = radar_df[clean_label(prov_b)].tolist()
            cats = categories + [categories[0]]
            a_vals_loop = a_vals + [a_vals[0]]
            b_vals_loop = b_vals + [b_vals[0]]

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatterpolar(
                r=a_vals_loop, theta=cats, fill='toself', name=clean_label(prov_a),
                line=dict(color=COLORS["green"], width=2.5),
                hovertemplate="<b>%{theta}</b><br>Skor: %{r:.1f}<extra></extra>"
            ))
            fig_cmp.add_trace(go.Scatterpolar(
                r=b_vals_loop, theta=cats, fill='toself', name=clean_label(prov_b),
                line=dict(color=COLORS["blue"], width=2.5),
                hovertemplate="<b>%{theta}</b><br>Skor: %{r:.1f}<extra></extra>"
            ))
            fig_cmp.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.08)"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.08)")
                ),
                legend=dict(orientation="h", y=-0.15),
                height=420,
                margin=dict(t=30, b=40, l=20, r=20),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)
            st.caption("Skor radar = normalisasi 0–100 per indikator pada tahun yang sama. Nilai mentah ada di tabel bawah.")

            st.dataframe(
                raw_df.style.format({
                    clean_label(prov_a): "{:,.2f}",
                    clean_label(prov_b): "{:,.2f}",
                }),
                use_container_width=True,
                height=220,
            )

# =========================================================
# TAB: NERACA PERDAGANGAN
# =========================================================
def render_trade():
    years = years_of(neraca)
    with st.container():
        st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
        yr_ner = st.selectbox("📅 Tahun", years[::-1], key="ner_yr")
        st.markdown('</div>', unsafe_allow_html=True)

    df_n_all = neraca[neraca["Tahun"] == yr_ner]
    tot_eks = df_n_all["Total_Ekspor"].sum()
    tot_imp = df_n_all["Total_Impor"].sum()
    net_all = tot_eks - tot_imp

    prev_year = yr_ner - 1
    df_prev = neraca[neraca["Tahun"] == prev_year]
    if not df_prev.empty:
        prev_eks = df_prev["Total_Ekspor"].sum()
        prev_imp = df_prev["Total_Impor"].sum()
        prev_net = prev_eks - prev_imp
    else:
        prev_eks = prev_imp = prev_net = None

    unit = "Jt"   # <--- DEFINE UNIT HERE

    def kpi_card(title, value, current, previous, unit="", reverse=False):
        # YoY calculation
        if previous is not None and previous != 0:
            yoy = (current - previous) / previous * 100
            arrow = "▲" if yoy > 0 else "▼"
            color = "green" if (yoy > 0) ^ reverse else "red"
            sign = "+" if yoy > 0 else "-"
            yoy_text = f"{arrow} {sign}{abs(yoy):.2f}%"
        else:
            yoy_text = "—"
            color = "gray"

        return f"""
        <div class="kpi-card" style="--card-accent:{color}">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1px;">
                <div style="font-size: 0.65rem; color: var(--muted);">YoY vs tahun lalu</div>
                <div style="font-size: 0.9rem; font-weight: 600; color: {color};">{yoy_text}</div>
            </div>
        </div>
        """

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(kpi_card("Total Ekspor", f"${tot_eks:,.0f} {unit}", tot_eks, prev_eks), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card("Total Impor", f"${tot_imp:,.0f} {unit}", tot_imp, prev_imp), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card("Net Trade", f"${net_all:,.0f} {unit}", net_all, prev_net), unsafe_allow_html=True)
    with k4:
        ratio_now = tot_eks / tot_imp if tot_imp != 0 else 0
        ratio_prev = (prev_eks / prev_imp) if prev_eks and prev_imp else None
        st.markdown(kpi_card("Rasio Ekspor/Impor", f"{ratio_now:.2f}x", ratio_now, ratio_prev, unit=""), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # DKI explanation
    dki = neraca[(neraca["Provinsi"]=="DKI JAKARTA")&(neraca["Tahun"]==yr_ner)]["Total_Impor"]
    if not dki.empty and float(dki.values[0]) > 80000:
        st.info(f"ℹ️ DKI Jakarta mencatat impor **${float(dki.values[0]):,.0f} Jt USD** — konsisten setiap tahun karena perannya sebagai hub distribusi impor nasional, bukan anomali data. Gunakan filter outlier untuk menyesuaikan skala.", icon="🏙️")

    df_n = df_n_all.copy()

    col_n1, col_n2 = st.columns(2)
    with col_n1:
        sec(f"Top 10 Provinsi Surplus — {yr_ner}", "", COLORS["green"])
        surplus = df_n[df_n["Net_Trade"]>0].nlargest(10,"Net_Trade").copy()
        surplus["label"] = surplus["Provinsi"].map(clean_label)
        if not surplus.empty:
            fig_ns = go.Figure(go.Bar(
                x=surplus["Net_Trade"], y=surplus["label"], orientation="h",
                marker=dict(color=surplus["Net_Trade"], colorscale=[[0,"#0d6b4a"],[1,"#22d3a4"]], showscale=False),
                hovertemplate="<b>%{y}</b><br>Net Trade: <b>$%{x:,.1f}M</b><br><i>Surplus — ekspor > impor</i><extra></extra>",
                text=surplus["Net_Trade"].apply(lambda v: f"${v:,.0f}M"),
                textposition="outside", textfont=dict(size=9, color="#9aa5be")
            ))
            fig_ns.update_yaxes(categoryorder="total ascending")
            apply_layout(fig_ns, h=330, xaxis=dict(title="Juta USD",gridcolor="rgba(255,255,255,0.04)"),yaxis=dict(title=""))
            st.plotly_chart(fig_ns, use_container_width=True)

    with col_n2:
        sec(f"Top 10 Provinsi Defisit — {yr_ner}", "", COLORS["red"])
        defisit = df_n[df_n["Net_Trade"]<0].nsmallest(10,"Net_Trade").copy()
        defisit["label"] = defisit["Provinsi"].map(clean_label)
        if not defisit.empty:
            fig_nd = go.Figure(go.Bar(
                x=defisit["Net_Trade"], y=defisit["label"], orientation="h",
                marker=dict(color=defisit["Net_Trade"], colorscale=[[0,"#f87171"],[1,"#450a0a"]], showscale=False),
                hovertemplate="<b>%{y}</b><br>Net Trade: <b>$%{x:,.1f}M</b><br><i>Defisit — impor > ekspor</i><extra></extra>",
                text=defisit["Net_Trade"].apply(lambda v: f"${v:,.0f}M"),
                textposition="outside", textfont=dict(size=9, color="#9aa5be")
            ))
            fig_nd.update_yaxes(categoryorder="total descending")
            apply_layout(fig_nd, h=330, xaxis=dict(title="Juta USD",gridcolor="rgba(255,255,255,0.04)"),yaxis=dict(title=""))
            st.plotly_chart(fig_nd, use_container_width=True)

    # Grouped bar ekspor vs impor
    sec(f"Ekspor vs Impor — Top 20 Provinsi · {yr_ner}", "Hover untuk breakdown Migas/Non-Migas")
    df_top20 = df_n.nlargest(20,"Total_Ekspor").copy()
    df_top20["label"] = df_top20["Provinsi"].map(clean_label)
    fig_gb = go.Figure()
    fig_gb.add_trace(go.Bar(name="Ekspor", x=df_top20["label"], y=df_top20["Total_Ekspor"],
        marker_color=COLORS["green"],
        hovertemplate="<b>%{x}</b><br>Ekspor Total: <b>$%{y:,.1f}M</b><br>Migas: $%{customdata[0]:,.1f}M · Non-Migas: $%{customdata[1]:,.1f}M<extra></extra>",
        customdata=df_top20[["Ekspor_Migas","Ekspor_NonMigas"]].values))
    fig_gb.add_trace(go.Bar(name="Impor", x=df_top20["label"], y=df_top20["Total_Impor"],
        marker_color=COLORS["red"],
        hovertemplate="<b>%{x}</b><br>Impor Total: <b>$%{y:,.1f}M</b><br>Migas: $%{customdata[0]:,.1f}M · Non-Migas: $%{customdata[1]:,.1f}M<extra></extra>",
        customdata=df_top20[["Impor_Migas","Impor_NonMigas"]].values))
    apply_layout(fig_gb, h=370, barmode="group",
                 xaxis=dict(tickangle=-35,title="",gridcolor="rgba(255,255,255,0.04)"),
                 yaxis=dict(title="Juta USD",gridcolor="rgba(255,255,255,0.04)"),
                 legend=dict(orientation="h",y=1.08))
    st.plotly_chart(fig_gb, use_container_width=True)

    # Donut Migas vs NonMigas
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        sec("Komposisi Ekspor", f"Migas vs Non-Migas · {yr_ner}")
        fig_d1 = go.Figure(go.Pie(
            labels=["Ekspor Migas","Ekspor Non-Migas"],
            values=[df_n_all["Ekspor_Migas"].sum(), df_n_all["Ekspor_NonMigas"].sum()],
            hole=0.5, marker=dict(colors=[COLORS["orange"],COLORS["green"]]),
            hovertemplate="<b>%{label}</b><br>$%{value:,.1f}M (%{percent})<extra></extra>",
            textinfo="percent+label", textfont=dict(size=11)
        ))
        fig_d1.update_layout(height=260, **{k:v for k,v in PLOT_LAYOUT.items() if k not in ["xaxis","yaxis","margin"]}, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_d1, use_container_width=True)

    with col_d2:
        sec("Komposisi Impor", f"Migas vs Non-Migas · {yr_ner}")
        fig_d2 = go.Figure(go.Pie(
            labels=["Impor Migas","Impor Non-Migas"],
            values=[df_n_all["Impor_Migas"].sum(), df_n_all["Impor_NonMigas"].sum()],
            hole=0.5, marker=dict(colors=[COLORS["red"],COLORS["purple"]]),
            hovertemplate="<b>%{label}</b><br>$%{value:,.1f}M (%{percent})<extra></extra>",
            textinfo="percent+label", textfont=dict(size=11)
        ))
        fig_d2.update_layout(height=260, **{k:v for k,v in PLOT_LAYOUT.items() if k not in ["xaxis","yaxis","margin"]}, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_d2, use_container_width=True)

    # Sankey flow
    st.markdown("<br>", unsafe_allow_html=True)
    sec("Sankey Flow — Ekspor & Impor per Provinsi", f"Top 10 · {yr_ner}")
    try:
        top10_trade = df_n.sort_values("Net_Trade", ascending=False).head(10).copy()
        top10_trade["label"] = top10_trade["Provinsi"].map(clean_label)
        nodes = ["Ekspor","Impor"] + top10_trade["label"].tolist()
        source, target, value, link_colors = [], [], [], []
        for i, row in enumerate(top10_trade.itertuples()):
            idx = i + 2
            source.extend([0, 1]); target.extend([idx, idx])
            value.extend([max(float(row.Total_Ekspor),0), max(float(row.Total_Impor),0)])
            link_colors.extend(["rgba(34,211,164,.35)","rgba(248,113,113,.35)"])
        sankey = go.Figure(go.Sankey(
            node=dict(pad=15, thickness=18,
                      line=dict(color="rgba(255,255,255,.15)", width=1),
                      label=nodes,
                      color=[COLORS["green"],COLORS["red"]] + ["rgba(255,255,255,.18)"]*len(top10_trade)),
            link=dict(source=source, target=target, value=value, color=link_colors)
        ))
        apply_layout(sankey, h=380)
        st.plotly_chart(sankey, use_container_width=True)
    except Exception as e:
        st.caption(f"Sankey tidak dapat dirender: {e}")

    # Tren net trade
    sec("Tren Net Trade — Provinsi Terpilih", "Garis putus = breakeven")
    prov_ner = st.multiselect("Pilih Provinsi",
        sorted([p for p in neraca["Provinsi"].unique() if p.upper() not in {"DKI JAKARTA"}]),
        default=[p for p in ["JAWA BARAT","RIAU","KALIMANTAN TIMUR","SULAWESI TENGAH"]
                 if p in neraca["Provinsi"].unique()],
        key="prov_ner")
    incl_dki = st.checkbox("Sertakan DKI Jakarta", value=False, key="ner_dki")
    if incl_dki: prov_ner = list(prov_ner) + ["DKI JAKARTA"]
    if prov_ner:
        df_nt = neraca[neraca["Provinsi"].isin(prov_ner)].copy()
        df_nt["label"] = df_nt["Provinsi"].map(clean_label)
        fig_nt = px.line(df_nt, x="Tahun", y="Net_Trade", color="label", markers=True,
                         color_discrete_sequence=PROV_PALETTE, labels={"Net_Trade":"Net Trade (Juta USD)","label":""})
        fig_nt.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Tahun: %{x}<br>Net Trade: <b>$%{y:,.1f}M</b><extra></extra>",
                             line_width=2.2, marker_size=5)
        fig_nt.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_dash="dot",
                         annotation_text="Breakeven", annotation_font=dict(color="#9aa5be",size=10))
        apply_layout(fig_nt, h=310, legend_h=True, yaxis=dict(title="Juta USD",gridcolor="rgba(255,255,255,0.04)"))
        st.plotly_chart(fig_nt, use_container_width=True)

    if not surplus.empty:
        insight_callout(f"Insight Neraca {yr_ner}", [
            f"Surplus terbesar: <b>{surplus.nlargest(1,'Net_Trade')['label'].values[0]}</b>",
            f"Net trade nasional: <b>{'Surplus' if net_all>=0 else 'Defisit'} ${abs(net_all):,.0f} Jt</b>",
            f"Rasio ekspor/impor: <b>{tot_eks/tot_imp:.2f}x</b> {'✅' if tot_eks>tot_imp else '⚠️'}",
            "Data 2025 = kumulatif tahunan sesuai laporan BPS. DKI Jakarta secara struktural selalu defisit besar sebagai hub impor.",
        ], tone="info")

# =========================================================
# TAB: PENDUDUK
# =========================================================
def render_population():
    years = years_of(penduduk)
    with st.container():
        st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns([2,2,2])
        with fc1: yr_pop = st.selectbox("📅 Tahun", years[::-1], key="pop_yr")
        with fc2: sort_by = st.selectbox("Urutkan", ["Jumlah Penduduk","Kepadatan","Laju Pertumbuhan"], key="pop_sort")
        with fc3: top_n = st.slider("Top N provinsi", 5, 38, 15, key="pop_n")
        st.markdown('</div>', unsafe_allow_html=True)

    df_pop = non_country(penduduk)[penduduk["Tahun"]==yr_pop].copy()
    pop_id = penduduk[(penduduk["Provinsi"].str.upper()=="INDONESIA")&(penduduk["Tahun"]==yr_pop)]

    v_pop  = float(pop_id["Jumlah_Penduduk_Ribu"].dropna().iloc[0]) if not pop_id.empty else None
    v_dens = float(pop_id["Kepadatan_per_Km2"].dropna().iloc[0]) if not pop_id.empty else None
    v_grow = float(pop_id["Laju_Pertumbuhan_Persen"].dropna().iloc[0]) if not pop_id.empty else None

    p1,p2,p3,p4 = st.columns(4)
    p1.metric("Total Penduduk Indonesia", f"{v_pop/1000:.1f} Juta jiwa" if v_pop else "—")
    p2.metric("Kepadatan Nasional", f"{v_dens:,.0f} jiwa/km²" if v_dens else "—")
    p3.metric("Laju Pertumbuhan", f"{v_grow:.2f}%/tahun" if v_grow else "—")
    if not df_pop.empty:
        p4.metric("Provinsi Terpadat", df_pop.nlargest(1,"Kepadatan_per_Km2")["Provinsi"].values[0])

    st.markdown("<br>", unsafe_allow_html=True)

    sort_col = {"Jumlah Penduduk":"Jumlah_Penduduk_Ribu","Kepadatan":"Kepadatan_per_Km2",
                "Laju Pertumbuhan":"Laju_Pertumbuhan_Persen"}[sort_by]

    col_a, col_b = st.columns(2)
    with col_a:
        sec("Distribusi Penduduk per Provinsi", f"Treemap · {yr_pop}")
        fig_tree = px.treemap(df_pop.dropna(subset=["Jumlah_Penduduk_Ribu"]),
            path=["Provinsi"], values="Jumlah_Penduduk_Ribu",
            color="Jumlah_Penduduk_Ribu",
            color_continuous_scale=[[0,"#042C53"],[0.5,"#378ADD"],[1,"#B5D4F4"]],
            custom_data=["Kepadatan_per_Km2","Laju_Pertumbuhan_Persen","Persentase_Penduduk"])
        fig_tree.update_traces(hovertemplate=(
            "<b>%{label}</b><br>Penduduk: <b>%{value:,.0f} Ribu</b><br>"
            "Kepadatan: <b>%{customdata[0]:,.0f} jiwa/km²</b><br>"
            "Pertumbuhan: <b>%{customdata[1]:.2f}%</b><br>"
            "Porsi: <b>%{customdata[2]:.1f}%</b><extra></extra>"))
        fig_tree.update_layout(margin=dict(t=10,b=0,l=0,r=0), height=380,
            paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
        st.plotly_chart(fig_tree, use_container_width=True)

    with col_b:
        sec(f"Top {top_n} Provinsi — {sort_by}", f"{yr_pop}")
        top_pop = df_pop.dropna(subset=[sort_col]).nlargest(top_n, sort_col).copy()
        top_pop["label"] = top_pop["Provinsi"].map(clean_label)
        fig_bar = go.Figure(go.Bar(
            x=top_pop[sort_col], y=top_pop["label"], orientation="h",
            marker=dict(color=top_pop[sort_col],
                        colorscale=[[0,"#042C53"],[0.5,"#2563EB"],[1,"#60a5fa"]], showscale=False),
            hovertemplate=(f"<b>%{{y}}</b><br>{sort_by}: <b>%{{x:,.1f}}</b><extra></extra>"),
            text=top_pop[sort_col].apply(lambda v: f"{v/1000:.1f}Jt" if (v>=1000 and sort_by=="Jumlah Penduduk") else f"{v:,.0f}"),
            textposition="outside", textfont=dict(size=9, color="#9aa5be")
        ))
        fig_bar.update_yaxes(categoryorder="total ascending")
        apply_layout(fig_bar, h=380, xaxis=dict(title=sort_by, gridcolor="rgba(255,255,255,0.04)"), yaxis=dict(title=""))
        st.plotly_chart(fig_bar, use_container_width=True)

    # Scatter kuadran kepadatan vs pertumbuhan
    sec("Kepadatan vs Laju Pertumbuhan", f"Bubble = jumlah penduduk · {yr_pop} · 4 kuadran")
    sc_pop = df_pop.dropna(subset=["Kepadatan_per_Km2","Laju_Pertumbuhan_Persen","Jumlah_Penduduk_Ribu"]).copy()
    sc_pop["label"] = sc_pop["Provinsi"].map(clean_label)
    avg_dens = sc_pop["Kepadatan_per_Km2"].mean()
    avg_grow = sc_pop["Laju_Pertumbuhan_Persen"].mean()
    fig_sc = px.scatter(sc_pop, x="Kepadatan_per_Km2", y="Laju_Pertumbuhan_Persen",
        size="Jumlah_Penduduk_Ribu", hover_name="label",
        color="Kepadatan_per_Km2",
        color_continuous_scale=[[0,"#1e3a5f"],[1,"#60a5fa"]], size_max=45,
        custom_data=["Jumlah_Penduduk_Ribu","Persentase_Penduduk","Rasio_Jenis_Kelamin"],
        labels={"Kepadatan_per_Km2":"Kepadatan (jiwa/km²)","Laju_Pertumbuhan_Persen":"Laju Pertumbuhan (%/tahun)"})
    fig_sc.update_traces(hovertemplate=(
        "<b>%{hovertext}</b><br>"
        "Kepadatan: <b>%{x:,.0f} jiwa/km²</b><br>"
        "Pertumbuhan: <b>%{y:.2f}%/tahun</b><br>"
        "Penduduk: <b>%{customdata[0]:,.0f} Ribu</b><br>"
        "Porsi nasional: <b>%{customdata[1]:.1f}%</b><br>"
        "Rasio jenis kelamin: <b>%{customdata[2]:.1f}</b><extra></extra>"))
    fig_sc.add_vline(x=avg_dens, line_dash="dot", line_color="rgba(255,255,255,0.15)",
                     annotation_text="Rata-rata kepadatan", annotation_font=dict(color="#6b7a99",size=9))
    fig_sc.add_hline(y=avg_grow, line_dash="dot", line_color="rgba(255,255,255,0.15)",
                     annotation_text="Rata-rata pertumbuhan", annotation_font=dict(color="#6b7a99",size=9))
    apply_layout(fig_sc, h=430,
        xaxis=dict(title="Kepadatan (jiwa/km²)", gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(title="Laju Pertumbuhan (%/tahun)", gridcolor="rgba(255,255,255,0.04)"),
        coloraxis=dict(colorbar=dict(thickness=10, len=0.6,
            title=dict(text="Kepadatan", font=dict(color="#9aa5be")), tickfont=dict(color="#9aa5be"))))
    st.plotly_chart(fig_sc, use_container_width=True)

    insight_callout("Insight 4 Kuadran Kepadatan vs Pertumbuhan", [
        "<b>Kanan atas</b>: padat & tumbuh cepat — tekanan urban tinggi (mis. Sulawesi Tengah)",
        "<b>Kiri atas</b>: jarang tapi tumbuh cepat — frontier/ekspansif (mis. Papua)",
        "<b>Kanan bawah</b>: padat tapi pertumbuhan melambat — matur (mis. DKI Jakarta)",
        "<b>Kiri bawah</b>: jarang & lambat — perlu perhatian kebijakan",
    ], tone="info")

    # Tabel
    sec("Tabel Lengkap", f"{yr_pop}")
    cols_show = [c for c in ["Provinsi","Jumlah_Penduduk_Ribu","Laju_Pertumbuhan_Persen",
                              "Persentase_Penduduk","Kepadatan_per_Km2","Rasio_Jenis_Kelamin"]
                 if c in df_pop.columns]
    df_show = df_pop[cols_show].copy()
    df_show["Provinsi"] = df_show["Provinsi"].map(clean_label)
    df_show = df_show.sort_values(sort_col, ascending=False).reset_index(drop=True)
    df_show.index += 1
    fmt_m = {"Jumlah_Penduduk_Ribu":"{:,.1f}","Laju_Pertumbuhan_Persen":"{:.2f}%",
             "Persentase_Penduduk":"{:.2f}%","Kepadatan_per_Km2":"{:,.0f}","Rasio_Jenis_Kelamin":"{:.1f}"}
    st.dataframe(df_show.style
        .background_gradient(cmap="Blues", subset=[c for c in ["Jumlah_Penduduk_Ribu"] if c in df_show.columns])
        .background_gradient(cmap="YlOrBr", subset=[c for c in ["Kepadatan_per_Km2"] if c in df_show.columns])
        .format({k:v for k,v in fmt_m.items() if k in df_show.columns}),
        use_container_width=True, height=420)

# =========================================================
# TAB: FORECAST
# =========================================================
def render_forecast():
    st.markdown("### 🔮 Forecasting Multi-Indicator")

    prov_list = get_prov_list(pdrb)
    indicator_list = list(INDICATOR_CONFIG.keys())

    with st.container():
        st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([2.5, 2.5, 1.1])
        with c1:
            prov_sel = st.selectbox("🏙️ Provinsi", prov_list, index=0, key="fc_prov")
        with c2:
            ind_sel = st.selectbox("📊 Indikator", indicator_list, key="fc_ind")
        with c3:
            horizon = st.selectbox("⏭️ Horizon", [3, 5, 7], index=1, key="fc_horizon")
        st.markdown('</div>', unsafe_allow_html=True)

    base_df = get_indicator_base_df(ind_sel)
    if base_df.empty:
        st.warning("Data indikator ini belum tersedia.")
        return

    dfp = base_df[base_df["Provinsi"] == prov_sel].copy().sort_values("Tahun")
    if dfp.empty:
        st.warning("Tidak ada data untuk provinsi yang dipilih.")
        return

    fc = forecast_linear(dfp, horizon=horizon)
    if fc.empty:
        st.warning("Data historis terlalu sedikit untuk forecasting.")
        return

    cfg = INDICATOR_CONFIG[ind_sel]
    title = cfg["title"]

    hist = fc[fc["type"] == "Actual"].copy()
    fut = fc[fc["type"] == "Forecast"].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["Tahun"], y=hist["value"],
        mode="lines+markers", name="Actual",
        line=dict(width=3, color=COLORS["green"]),
        marker=dict(size=6),
        hovertemplate="<b>%{x}</b><br>Aktual: <b>%{y:.2f}</b><extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=fut["Tahun"], y=fut["pred"],
        mode="lines+markers", name="Forecast",
        line=dict(width=3, dash="dash", color=COLORS["orange"]),
        marker=dict(size=6),
        hovertemplate="<b>%{x}</b><br>Prediksi: <b>%{y:.2f}</b><extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([fut["Tahun"], fut["Tahun"][::-1]]),
        y=pd.concat([fut["upper"], fut["lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(251,146,60,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence band",
        hoverinfo="skip",
        showlegend=True
    ))

    apply_layout(fig, h=420, legend_h=True, yaxis=dict(title=title, gridcolor="rgba(255,255,255,0.04)"))
    sec(f"Forecast — {title}", f"{clean_label(prov_sel)}")
    st.plotly_chart(fig, use_container_width=True)

    last_actual = float(hist["value"].iloc[-1])
    last_pred = float(fut["pred"].iloc[-1])
    delta_pct = ((last_pred - last_actual) / last_actual * 100) if last_actual not in [0, None] else np.nan

    insight_callout("Insight Forecast", [
        f"Nilai terakhir aktual: <b>{last_actual:.2f}</b>",
        f"Prediksi tahun ke-{int(fut['Tahun'].iloc[-1])}: <b>{last_pred:.2f}</b>",
        f"Perubahan terhadap data terakhir: <b>{delta_pct:+.2f}%</b>",
        f"Jumlah titik historis: <b>{len(hist)}</b>",
    ], tone="good" if pd.notna(delta_pct) and delta_pct >= 0 else "warn")

    st.dataframe(
        fc[["Tahun", "type", "value", "pred", "lower", "upper"]].rename(columns={
            "value": "Aktual",
            "pred": "Prediksi",
            "lower": "Lower CI",
            "upper": "Upper CI",
            "type": "Tipe"
        }),
        use_container_width=True,
        height=280,
    )

# =========================================================
# TAB: AI ANALYTICS
# =========================================================
def build_cluster_frame(latest_year: int):
    p_cl = non_country(pdrb)[pdrb["Tahun"] == latest_year][["Provinsi", "PDRB_PerKapita_RibuRupiah"]]
    t_cl = non_country(tpt)[(non_country(tpt)["Periode"] == "Agustus") & (non_country(tpt)["Tahun"] == latest_year)][["Provinsi", "TPT_Persen"]]
    m_cl = non_country(miskin)[(non_country(miskin)["Daerah"] == "Jumlah") & (non_country(miskin)["Semester"] == "Semester 1 (Maret)") & (non_country(miskin)["Tahun"] == latest_year)][["Provinsi", "Persen_Penduduk_Miskin"]]
    g_cl = non_country(gini)[(non_country(gini)["Daerah"] == "Perkotaan+Perdesaan") & (non_country(gini)["Semester"] == "Semester 1 (Maret)") & (non_country(gini)["Tahun"] == latest_year)][["Provinsi", "Gini_Ratio"]]

    cl_df = p_cl.merge(t_cl, on="Provinsi", how="inner").merge(m_cl, on="Provinsi", how="inner").merge(g_cl, on="Provinsi", how="inner")
    if cl_df.empty:
        return cl_df, None, None, None, None, None

    cl_df["label"] = cl_df["Provinsi"].map(clean_label)
    cl_df["PDRB_log"] = np.log1p(cl_df["PDRB_PerKapita_RibuRupiah"])

    feats = ["PDRB_log", "TPT_Persen", "Persen_Penduduk_Miskin", "Gini_Ratio"]
    X = cl_df[feats].copy().fillna(cl_df[feats].median(numeric_only=True))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    best_k, best_score, best_labels = 3, -1, None
    max_k = min(6, len(cl_df) - 1)
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=25)
        labels = km.fit_predict(Xs)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(Xs, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    if best_labels is None:
        best_labels = KMeans(n_clusters=3, random_state=42, n_init=25).fit_predict(Xs)
        best_k = 3
        best_score = silhouette_score(Xs, best_labels) if len(set(best_labels)) > 1 else np.nan

    cl_df["cluster_id"] = best_labels.astype(int)
    profile = cl_df.groupby("cluster_id")[ ["PDRB_PerKapita_RibuRupiah", "TPT_Persen", "Persen_Penduduk_Miskin", "Gini_Ratio"] ].mean()
    order = profile["PDRB_PerKapita_RibuRupiah"].sort_values(ascending=False).index.tolist()
    cluster_map = {cid: f"Cluster {chr(65+i)}" for i, cid in enumerate(order)}
    cl_df["cluster_name"] = cl_df["cluster_id"].map(cluster_map)

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(Xs)
    cl_df["PC1"] = pcs[:, 0]
    cl_df["PC2"] = pcs[:, 1]

    return cl_df, feats, pca, best_k, best_score, scaler


def render_ai():
    years = years_of(pdrb)
    latest_year = max(years)
    sec("Forecast Ringkas", f"Basis {latest_year}")

    # simple forecast using linear trend for PDRB
    nat_p = pdrb[pdrb["Provinsi"].str.upper() == "INDONESIA"].sort_values("Tahun")
    nat_p = nat_p[["Tahun", "PDRB_PerKapita_RibuRupiah"]].dropna()
    if len(nat_p) >= 3:
        x = nat_p["Tahun"].astype(int).to_numpy()
        y = nat_p["PDRB_PerKapita_RibuRupiah"].to_numpy(dtype=float)
        coef = np.polyfit(x, y, 1)
        future_years = np.array([latest_year + i for i in range(1, 6)])
        pred = np.polyval(coef, future_years)
        fdf = pd.DataFrame({"Tahun": future_years, "Prediksi_PDRB": pred})
        fig_f = px.line(pd.concat([nat_p.rename(columns={"PDRB_PerKapita_RibuRupiah": "Nilai"}), fdf.rename(columns={"Prediksi_PDRB": "Nilai"})], ignore_index=True), x="Tahun", y="Nilai")
        fig_f.add_vline(x=latest_year, line_dash="dot", line_color="rgba(255,255,255,0.35)")
        apply_layout(fig_f, h=300, yaxis=dict(title="Rp Ribu", gridcolor="rgba(255,255,255,0.04)"))
        st.plotly_chart(fig_f, use_container_width=True)
        insight_callout("Forecast sederhana", [f"Proyeksi 5 tahun ke depan berdasarkan tren historis.", f"Prediksi {future_years[-1]}: <b>Rp {pred[-1]:,.0f} Ribu</b>"], tone="warn")

    st.markdown("<br>", unsafe_allow_html=True)
    sec("Clustering Provinsi — K-Means Otomatis", "PCA 2D + silhouette selection")
    result = build_cluster_frame(latest_year)
    cl_df, feats, pca, best_k, best_score, scaler = result

    if cl_df is None or cl_df.empty:
        st.warning("Data clustering tidak tersedia.")
        return

    fig_cl = px.scatter(
        cl_df,
        x="PC1",
        y="PC2",
        color="cluster_name",
        hover_name="label",
        size="PDRB_PerKapita_RibuRupiah",
        color_discrete_sequence=[COLORS["green"], COLORS["orange"], COLORS["red"], COLORS["blue"], COLORS["purple"]],
        labels={"PC1": f"PC1 (explained {pca.explained_variance_ratio_[0] * 100:.1f}%)", "PC2": f"PC2 (explained {pca.explained_variance_ratio_[1] * 100:.1f}%)", "cluster_name": "Cluster"}
    )
    fig_cl.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Cluster: %{fullData.name}<br>"
            "PDRB: Rp %{customdata[0]:,.0f} Ribu<br>"
            "TPT: %{customdata[1]:.2f}%<br>"
            "Kemiskinan: %{customdata[2]:.2f}%<br>"
            "Gini: %{customdata[3]:.3f}<extra></extra>"
        ),
        customdata=np.stack([
            cl_df["PDRB_PerKapita_RibuRupiah"],
            cl_df["TPT_Persen"],
            cl_df["Persen_Penduduk_Miskin"],
            cl_df["Gini_Ratio"],
        ], axis=-1)
    )
    apply_layout(fig_cl, h=420, legend_h=True, xaxis=dict(title=f"PC1 (explained {pca.explained_variance_ratio_[0] * 100:.1f}%)"), yaxis=dict(title=f"PC2 (explained {pca.explained_variance_ratio_[1] * 100:.1f}%)"))
    st.plotly_chart(fig_cl, use_container_width=True)
    st.caption(f"Silhouette terbaik: k={best_k} | score={best_score:.3f}")

    cluster_profile = cl_df.groupby("cluster_name")[ ["PDRB_PerKapita_RibuRupiah", "TPT_Persen", "Persen_Penduduk_Miskin", "Gini_Ratio"] ].mean().reset_index()
    heat = cluster_profile.set_index("cluster_name")
    heat_z = heat.copy()
    for c in heat.columns:
        sd = heat[c].std()
        heat_z[c] = 0.0 if pd.isna(sd) or sd == 0 else (heat[c] - heat[c].mean()) / sd

    sec("Profil Cluster (standardized)", "z-score heatmap")
    fig_prof = px.imshow(
        heat_z,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-2,
        zmax=2,
        labels=dict(x="Indikator", y="Cluster", color="Z-score"),
    )
    fig_prof.update_layout(height=360, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_prof, use_container_width=True)

    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.dataframe(cl_df[["label", "cluster_name", "PDRB_PerKapita_RibuRupiah", "TPT_Persen", "Persen_Penduduk_Miskin", "Gini_Ratio"]].sort_values(["cluster_name", "PDRB_PerKapita_RibuRupiah"], ascending=[True, False]), use_container_width=True, height=300)
    with c2:
        st.markdown("#### Ringkasan Cluster")
        summary = cl_df.groupby("cluster_name")[ ["PDRB_PerKapita_RibuRupiah", "TPT_Persen", "Persen_Penduduk_Miskin", "Gini_Ratio"] ].mean().sort_values("PDRB_PerKapita_RibuRupiah", ascending=False)
        st.dataframe(summary.style.format("{:.2f}"), use_container_width=True, height=300)

# =========================================================
# TAB: LAINNYA
# =========================================================
def render_more():
    sub = st.radio("Pilih fitur", ["📊 Korelasi Matrix","⬇️ Export CSV","📖 Story Mode"],
                   horizontal=True, key="more_sub")

    if sub == "📊 Korelasi Matrix":
        sec("Correlation Matrix", "Multi-indikator · tahun terbaru")
        latest = max(years_of(pdrb))
        p_c = non_country(pdrb)[pdrb["Tahun"]==latest][["Provinsi","PDRB_PerKapita_RibuRupiah"]]
        t_c = non_country(tpt)[(non_country(tpt)["Periode"]=="Agustus")&(non_country(tpt)["Tahun"]==latest)][["Provinsi","TPT_Persen"]]
        m_c = non_country(miskin)[(non_country(miskin)["Daerah"]=="Jumlah")&(non_country(miskin)["Semester"]=="Semester 1 (Maret)")&(non_country(miskin)["Tahun"]==latest)][["Provinsi","Persen_Penduduk_Miskin"]]
        g_c = non_country(gini)[(non_country(gini)["Daerah"]=="Perkotaan+Perdesaan")&(non_country(gini)["Semester"]=="Semester 1 (Maret)")&(non_country(gini)["Tahun"]==latest)][["Provinsi","Gini_Ratio"]]
        df_corr = p_c.merge(t_c,on="Provinsi").merge(m_c,on="Provinsi").merge(g_c,on="Provinsi")
        if not df_corr.empty:
            corr_m = df_corr.drop(columns=["Provinsi"]).corr()
            fig_corr = px.imshow(corr_m, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1)
            fig_corr.update_traces(hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = <b>%{z:.3f}</b><extra></extra>")
            apply_layout(fig_corr, h=420)
            st.plotly_chart(fig_corr, use_container_width=True)
            insight_callout("Interpretasi Korelasi", [
                "Warna merah gelap = korelasi negatif kuat (variabel bergerak berlawanan)",
                "Warna biru gelap = korelasi positif kuat (variabel bergerak searah)",
                f"PDRB vs Kemiskinan: r = {corr_m.loc['PDRB_PerKapita_RibuRupiah','Persen_Penduduk_Miskin']:.3f} — {'negatif kuat ✅' if corr_m.loc['PDRB_PerKapita_RibuRupiah','Persen_Penduduk_Miskin'] < -0.5 else 'sedang'}",
            ], tone="info")

    elif sub == "⬇️ Export CSV":
        sec("Export Dataset", "Unduh data sesuai kebutuhan")
        export_choice = st.selectbox("Dataset", ["PDRB","Inflasi","Neraca Perdagangan",
                                                   "TPT","Kemiskinan","Gini Ratio","Penduduk"])
        ds_map = {"PDRB":pdrb,"Inflasi":inflasi,"Neraca Perdagangan":neraca,
                  "TPT":tpt,"Kemiskinan":miskin,"Gini Ratio":gini,"Penduduk":penduduk}
        df_exp = ds_map[export_choice]
        st.dataframe(df_exp.head(100), use_container_width=True)
        csv = df_exp.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇️ Download {export_choice}.csv",
            data=csv,
            file_name=f"{export_choice.lower().replace(' ','_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption(f"Total baris: {len(df_exp):,} · Total kolom: {len(df_exp.columns)}")

    else:  # Story Mode
        sec("Story Mode", "Narasi otomatis kondisi ekonomi Indonesia")
        latest = max(years_of(pdrb))
        nat_pdrb = pdrb[pdrb["Provinsi"].str.upper()=="INDONESIA"]
        v_p = float(nat_pdrb[nat_pdrb["Tahun"]==latest]["PDRB_PerKapita_RibuRupiah"].iloc[0]) if not nat_pdrb[nat_pdrb["Tahun"]==latest].empty else None
        top10_pdrb = non_country(pdrb)[pdrb["Tahun"]==latest].nlargest(3,"PDRB_PerKapita_RibuRupiah")["Provinsi"].map(clean_label).tolist()
        st.markdown(f"""
<div style='background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:20px 24px;line-height:1.8;font-size:14px;'>

### 🇮🇩 Narasi Ekonomi Indonesia — {latest}

**Pertumbuhan Ekonomi**

Indonesia terus menunjukkan pertumbuhan ekonomi yang solid. PDRB per kapita nasional mencapai
**Rp {v_p:,.0f} Ribu** pada {latest}, menandakan peningkatan kesejahteraan yang berkelanjutan.
Provinsi dengan kontribusi tertinggi antara lain: **{', '.join(top10_pdrb)}** — yang mencerminkan
dominasi ekonomi pulau Jawa dan Kalimantan dalam struktur PDB nasional.

**Ketenagakerjaan & Kemiskinan**

Tingkat Pengangguran Terbuka (TPT) nasional menunjukkan tren penurunan dalam beberapa tahun terakhir,
meskipun masih terdapat kesenjangan antarprovinsi yang signifikan, terutama antara kawasan barat dan timur Indonesia.
Program pengentasan kemiskinan terus memberikan hasil positif, namun akselerasi diperlukan di
provinsi-provinsi dengan kemiskinan struktural tinggi.

**Ketimpangan & Stabilitas Harga**

Gini Ratio nasional relatif stabil, mengindikasikan tidak adanya lonjakan ketimpangan yang dramatis.
Inflasi YoY terpantau dalam kisaran yang bisa diterima, sejalan dengan kebijakan moneter Bank Indonesia
yang berfokus pada stabilitas harga.

**Neraca Perdagangan**

Indonesia secara keseluruhan mencatat surplus perdagangan, didorong oleh ekspor komoditas
dari Sumatera, Kalimantan, dan Sulawesi. DKI Jakarta sebagai hub impor nasional
secara struktural mencatat defisit besar, yang merupakan cerminan perannya sebagai
pusat distribusi dan konsumsi, bukan indikator negatif.

---
*Narasi ini dibuat otomatis berdasarkan data BPS. Untuk analisis mendalam, gunakan tab Peta, Tren, dan AI Analytics.*
</div>
""", unsafe_allow_html=True)

# =========================================================
# ROUTING
# =========================================================
if st.session_state.active_tab == "summary":
    render_summary()
elif st.session_state.active_tab == "map":
    render_map()
elif st.session_state.active_tab == "trend":
    render_trend()
elif st.session_state.active_tab == "comparison":
    render_comparison()
elif st.session_state.active_tab == "trade":
    render_trade()
elif st.session_state.active_tab == "population":
    render_population()
elif st.session_state.active_tab == "forecast":
    render_forecast()
elif st.session_state.active_tab == "ai":
    render_ai()
else:
    render_more()