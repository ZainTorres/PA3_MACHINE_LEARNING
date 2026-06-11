"""
PA3 / Dashboard Scopus — ML aplicado a Detección de Ciberataques
Curso: Fundamentos de Machine Learning
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

# ── CONFIGURACIÓN ──
st.set_page_config(
    page_title="ML en Ciberseguridad — Scopus Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

GITHUB_RAW_URL = "https://raw.githubusercontent.com/ZainTorres/PA3_MACHINE_LEARNING/main/scopus_data.csv"

# ── ESTILOS ──
st.markdown("""
<style>
    .stApp { background-color: #0d1117; }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #1f2937; }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #111827, #1a2235);
        border: 1px solid #1f2937; border-radius: 12px; padding: 16px 20px;
    }
    [data-testid="stMetricValue"] { color: #38bdf8 !important; font-size: 1.9rem !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    h1 { color: #f1f5f9 !important; }
    h2, h3 { color: #cbd5e1 !important; }
    hr { border-color: #1f2937; }
    .insight-box {
        background-color: #111827;
        border-left: 4px solid #38bdf8;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px; margin: 6px 0 18px 0;
        color: #94a3b8; font-size: 0.9rem; line-height: 1.6;
    }
    .question-box {
        background: linear-gradient(135deg, #0f172a, #111827);
        border: 1px solid #38bdf8; border-radius: 14px;
        padding: 20px 28px; margin-bottom: 24px; text-align: center;
    }
    .question-box p { color: #e2e8f0; font-size: 1.05rem; margin: 0; line-height: 1.8; }
    .question-label {
        color: #38bdf8; font-size: 0.78rem; font-weight: 700;
        letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px;
    }
    .chip {
        display: inline-block;
        background: rgba(56,189,248,0.12);
        color: #7dd3fc; border: 1px solid #0ea5e9;
        border-radius: 20px; padding: 5px 14px;
        margin: 4px 4px 4px 0;
        font-family: monospace; font-size: 0.85rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

PAPER_BG = "#0d1117"
PLOT_BG  = "#111827"
COLORS   = ["#38bdf8","#818cf8","#34d399","#fb923c","#f472b6","#a78bfa","#facc15","#f87171"]


# ── HELPERS ──
@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    return _clean(df)

@st.cache_data
def load_github(url):
    df = pd.read_csv(url)
    return _clean(df)

def _clean(df):
    df.columns = df.columns.str.strip()
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    if "Cited by" in df.columns:
        df["Cited by"] = pd.to_numeric(df["Cited by"], errors="coerce").fillna(0).astype(int)
    return df

def top_authors(df, n=12):
    all_a = []
    for row in df["Authors"].dropna():
        all_a.extend([a.strip() for a in str(row).split(";")])
    c = Counter(all_a)
    return pd.DataFrame(c.most_common(n), columns=["Autor","Publicaciones"])

def top_keywords(df, n=20):
    all_k = []
    col = "Author Keywords" if "Author Keywords" in df.columns else None
    if not col:
        return pd.DataFrame()
    for row in df[col].dropna():
        all_k.extend([k.strip().lower().title() for k in str(row).split(";")])
    c = Counter(all_k)
    return pd.DataFrame(c.most_common(n), columns=["Keyword","Frecuencia"])

def wordfreq(df, n=30):
    stop = {
        "the","a","an","and","or","of","in","to","is","are","was","were","this",
        "that","for","on","with","as","by","it","be","from","at","has","have",
        "been","their","they","which","also","can","not","more","its","we","our",
        "these","such","both","may","used","using","based","results","study",
        "paper","data","model","new","two","three","one","all","however","found",
        "show","showed","significant","participants","reported","higher","lower",
    }
    text = " ".join(df["Abstract"].dropna().astype(str)).lower()
    words = [w for w in re.findall(r"\b[a-z]{5,}\b", text) if w not in stop]
    c = Counter(words)
    return pd.DataFrame(c.most_common(n), columns=["Palabra","Frecuencia"])

def insight(text):
    st.markdown(
        f'<div class="insight-box">💡 <b>¿Qué nos dice este gráfico?</b><br>{text}</div>',
        unsafe_allow_html=True
    )


# ── SIDEBAR ──
with st.sidebar:
    st.markdown("## 📂 Fuente de datos")
    source = st.radio(
        "Cómo cargar el CSV:",
        ["✅ GitHub (automático)", "📁 Subir archivo local"]
    )
    df = None

    if source == "✅ GitHub (automático)":
        try:
            df = load_github(GITHUB_RAW_URL)
            st.success(f"✅ {len(df)} artículos cargados desde GitHub")
            st.markdown(f"[Ver CSV en GitHub]({GITHUB_RAW_URL})")
        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        up = st.file_uploader("Sube tu scopus_data.csv", type=["csv"])
        if up:
            df = load_csv(up)
            st.success(f"✅ {len(df)} artículos cargados")

    if df is not None and "Year" in df.columns:
        st.markdown("---")
        st.markdown("### 🎛️ Filtros")
        years = sorted(df["Year"].dropna().unique().astype(int))
        rng = st.slider("Rango de años", min(years), max(years), (min(years), max(years)))
        df = df[df["Year"].between(rng[0], rng[1])]
        if "Document Type" in df.columns:
            tipos = df["Document Type"].dropna().unique().tolist()
            sel = st.multiselect("Tipo de documento", tipos, default=tipos)
            if sel:
                df = df[df["Document Type"].isin(sel)]
        st.markdown(f"**Mostrando:** {len(df)} artículos")

    st.markdown("---")
    st.markdown("<small style='color:#475569'>PA3 · Fundamentos de Machine Learning · Ingeniería de Sistemas</small>", unsafe_allow_html=True)


# ── HEADER ──
st.markdown("""
<h1 style='text-align:center; padding:1rem 0 0.3rem;'>
🔐 Machine Learning en Ciberseguridad
</h1>
<p style='text-align:center; color:#64748b; font-size:0.95rem; margin-bottom:1.2rem;'>
Análisis bibliométrico · Scopus · 2024–2025
</p>
""", unsafe_allow_html=True)

st.markdown("""
<div class="question-box">
  <div class="question-label">🔬 Pregunta de investigación</div>
  <p>¿Cómo han evolucionado las técnicas de Machine Learning aplicadas a la detección
  de ciberataques en redes informáticas durante el período 2018–2025?</p>
</div>
""", unsafe_allow_html=True)

with st.expander("📌 Keywords utilizadas en Scopus", expanded=True):
    st.markdown("""
    <span class='chip'>machine learning</span>
    <span class='chip'>deep learning</span>
    <span class='chip'>intrusion detection</span>
    <span class='chip'>network security</span>
    <div style='margin-top:12px; background:#0f172a; border-radius:8px; padding:12px 16px;
         color:#64748b; font-size:0.88rem; line-height:1.6;'>
      Carga automática desde GitHub — o sube tu propio CSV exportado de Scopus desde el panel lateral.
      El dashboard detecta automáticamente columnas clave como autores, título, año, citas y keywords.
    </div>
    """, unsafe_allow_html=True)

if df is None:
    st.info("👈 Selecciona una fuente de datos en el panel lateral para comenzar.")
    st.stop()


# ── MÉTRICAS ──
st.markdown("---")
st.markdown("## 📊 Resumen del dataset")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📄 Artículos", len(df))
if "Cited by" in df.columns:
    c2.metric("🔖 Total citas", f"{int(df['Cited by'].sum()):,}")
    c3.metric("⭐ Máx. citas", int(df["Cited by"].max()))
if "Year" in df.columns:
    c4.metric("📅 Año más reciente", int(df["Year"].max()))
if "Source title" in df.columns:
    c5.metric("📰 Revistas únicas", df["Source title"].nunique())


# ── FILA 1: Publicaciones por año + Tipo doc ──
st.markdown("---")
st.markdown("## 📅 Tendencia de publicaciones")
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Publicaciones por año")
    if "Year" in df.columns:
        yc = df["Year"].value_counts().sort_index().reset_index()
        yc.columns = ["Año", "Publicaciones"]
        fig = px.bar(yc, x="Año", y="Publicaciones",
                     color_discrete_sequence=[COLORS[0]],
                     template="plotly_dark", text="Publicaciones")
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                          xaxis=dict(dtick=1), margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
        insight("El crecimiento sostenido refleja el auge del interés académico en ML para ciberseguridad. Artículos publicados en 2025 confirman que el campo sigue siendo altamente activo.")

with col2:
    st.markdown("### Tipo de documento")
    if "Document Type" in df.columns:
        dc = df["Document Type"].value_counts().reset_index()
        dc.columns = ["Tipo", "Cantidad"]
        fig = px.pie(dc, names="Tipo", values="Cantidad",
                     color_discrete_sequence=COLORS,
                     template="plotly_dark", hole=0.45)
        fig.update_layout(paper_bgcolor=PAPER_BG, margin=dict(t=20, b=20),
                          showlegend=True, legend=dict(font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)
        insight("La mayoría son artículos originales, lo que indica producción activa de evidencia empírica.")


# ── FILA 2: Más citados + Revistas ──
st.markdown("---")
st.markdown("## 🏆 Relevancia científica")
col3, col4 = st.columns(2)

with col3:
    st.markdown("### Artículos más citados")
    if "Cited by" in df.columns and "Title" in df.columns:
        top = df[["Title","Authors","Year","Cited by"]].nlargest(8, "Cited by").reset_index(drop=True)
        top["Título"] = top["Title"].str[:52] + "…"
        fig = px.bar(top, x="Cited by", y="Título", orientation="h",
                     color="Cited by",
                     color_continuous_scale=["#1e3a5f", "#38bdf8"],
                     template="plotly_dark",
                     hover_data={"Authors": True, "Year": True, "Título": False})
        fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                          yaxis=dict(autorange="reversed"),
                          coloraxis_showscale=False, margin=dict(t=20, b=20, l=10))
        st.plotly_chart(fig, use_container_width=True)
        insight("Los artículos más citados son las referencias obligatorias del marco teórico. Exploran modelos de IDS con Deep Learning y técnicas híbridas.")

with col4:
    st.markdown("### Revistas con más publicaciones")
    if "Source title" in df.columns:
        jc = df["Source title"].value_counts().head(8).reset_index()
        jc.columns = ["Revista", "Artículos"]
        jc["Revista corta"] = jc["Revista"].str[:42] + "…"
        fig = px.bar(jc, x="Artículos", y="Revista corta", orientation="h",
                     color_discrete_sequence=[COLORS[2]],
                     template="plotly_dark", text="Artículos")
        fig.update_traces(textposition="outside")
        fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                          yaxis=dict(autorange="reversed"),
                          margin=dict(t=20, b=20, l=10))
        st.plotly_chart(fig, use_container_width=True)
        insight("Las revistas más activas son referentes en Inteligencia Artificial aplicada a seguridad de redes.")


# ── FILA 3: Keywords + Treemap abstracts ──
st.markdown("---")
st.markdown("## 🏷️ Análisis de conceptos")
col5, col6 = st.columns(2)

with col5:
    st.markdown("### Keywords más frecuentes")
    kw_df = top_keywords(df, 18)
    if not kw_df.empty:
        fig = px.bar(kw_df, x="Frecuencia", y="Keyword", orientation="h",
                     color="Frecuencia",
                     color_continuous_scale=["#1e3a5f", "#facc15"],
                     template="plotly_dark")
        fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                          yaxis=dict(autorange="reversed"),
                          coloraxis_showscale=False, margin=dict(t=20, b=20, l=10))
        st.plotly_chart(fig, use_container_width=True)
        insight("Las keywords más repetidas revelan los ejes conceptuales: <b>Deep Learning</b>, <b>Intrusion Detection</b> y <b>Network Security</b> son el núcleo del campo.")

with col6:
    st.markdown("### Conceptos dominantes en abstracts")
    if "Abstract" in df.columns:
        wd = wordfreq(df, 25)
        if not wd.empty:
            fig = px.treemap(wd, path=["Palabra"], values="Frecuencia",
                             color="Frecuencia",
                             color_continuous_scale=["#111827", "#818cf8"],
                             template="plotly_dark")
            fig.update_layout(paper_bgcolor=PAPER_BG, margin=dict(t=20, b=20, l=10, r=10))
            fig.update_traces(textfont_size=13)
            st.plotly_chart(fig, use_container_width=True)
            insight("El tamaño de cada bloque es proporcional a su frecuencia. Los términos más grandes son los conceptos que la literatura discute con mayor profundidad.")


# ── FILA 4: Autores + Citas scatter ──
st.markdown("---")
st.markdown("## 👥 Comunidad investigadora e impacto")
col7, col8 = st.columns(2)

with col7:
    st.markdown("### Autores más productivos")
    auth_df = top_authors(df, 12)
    if not auth_df.empty:
        fig = px.bar(auth_df.sort_values("Publicaciones"),
                     x="Publicaciones", y="Autor", orientation="h",
                     color_discrete_sequence=[COLORS[4]],
                     template="plotly_dark", text="Publicaciones")
        fig.update_traces(textposition="outside")
        fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                          margin=dict(t=20, b=20, l=10))
        st.plotly_chart(fig, use_container_width=True)
        insight("Los autores más activos son los referentes ideales para citar en el estado del arte.")

with col8:
    st.markdown("### Citas por artículo y año")
    if "Year" in df.columns and "Cited by" in df.columns and "Title" in df.columns:
        fig = px.scatter(df, x="Year", y="Cited by",
                         size="Cited by", size_max=45,
                         color="Cited by",
                         color_continuous_scale=["#1e3a5f","#38bdf8"],
                         template="plotly_dark",
                         hover_data={"Title": True},
                         labels={"Year":"Año","Cited by":"Citas"})
        fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                          coloraxis_showscale=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
        insight("El artículo con más citas (burbuja más grande) es el trabajo más influyente del corpus. Los artículos de 2024 con ya muchas citas indican impacto inmediato.")


# ── TABLA ──
st.markdown("---")
st.markdown("## 📋 Explorador de artículos")
search = st.text_input("🔍 Buscar", placeholder="ej: CNN, LSTM, Random Forest, intrusion, XGBoost…")
cols_show = [c for c in ["Title","Authors","Year","Source title","Cited by","Author Keywords","DOI"] if c in df.columns]
df_show = df[cols_show].copy()
if search:
    mask = df["Title"].str.contains(search, case=False, na=False)
    if "Abstract" in df.columns:
        mask = mask | df["Abstract"].str.contains(search, case=False, na=False)
    df_show = df_show[mask]
    st.caption(f"🔎 {len(df_show)} resultado(s) para «{search}»")

if "DOI" in df_show.columns:
    df_show["DOI"] = df_show["DOI"].apply(
        lambda x: f"https://doi.org/{x}" if pd.notna(x) and str(x).strip() else ""
    )

st.dataframe(df_show.reset_index(drop=True), use_container_width=True, height=380,
    column_config={
        "DOI": st.column_config.LinkColumn("DOI", display_text="Abrir 🔗"),
        "Cited by": st.column_config.NumberColumn("Citas", format="%d"),
        "Year": st.column_config.NumberColumn("Año", format="%d"),
    })


# ── DESCARGA ──
st.markdown("---")
st.download_button(
    label="⬇️ Descargar CSV filtrado",
    data=df[cols_show].to_csv(index=False).encode("utf-8"),
    file_name="scopus_ml_ciberseguridad.csv",
    mime="text/csv",
    use_container_width=True
)

# ── FOOTER ──
st.markdown("""
<hr>
<p style='text-align:center; color:#334155; font-size:0.82rem;'>
Este aplicativo y su código fuente se distribuyen bajo la
<b>Apache License 2.0</b> · Dashboard desarrollado con Streamlit & Plotly ·
Fuente: Scopus (Elsevier) · Tema: Machine Learning aplicado a Ciberseguridad
</p>
""", unsafe_allow_html=True)
