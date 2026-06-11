"""
Dashboard de Literatura Académica - Scopus
Tema: Machine Learning aplicado a Detección de Ciberataques
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import re

# ── CONFIGURACIÓN ──
st.set_page_config(
    page_title="ML en Ciberseguridad | Scopus",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  .main-title { font-size:2.1rem; font-weight:800; color:#1a1a2e; text-align:center; padding:10px 0; }
  .subtitle   { font-size:1rem; color:#666; text-align:center; margin-bottom:25px; }
  .section-header { font-size:1.2rem; font-weight:700; color:#1a1a2e;
                    border-left:5px solid #667eea; padding-left:12px; margin:20px 0 10px 0; }
  .info-box { background:#f0f4ff; border-radius:10px; padding:15px 20px;
              border-left:4px solid #667eea; margin-bottom:20px; }
</style>
""", unsafe_allow_html=True)


# ── FUNCIONES ──
@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    if "Cited by" in df.columns:
        df["Cited by"] = pd.to_numeric(df["Cited by"], errors="coerce").fillna(0).astype(int)
    return df

@st.cache_data
def load_github(url):
    raw = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    df = pd.read_csv(raw)
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
    return pd.DataFrame(c.most_common(n), columns=["Autor", "Publicaciones"])

def top_keywords(df, n=20):
    all_k = []
    col = "Author Keywords" if "Author Keywords" in df.columns else None
    if not col:
        return pd.DataFrame()
    for row in df[col].dropna():
        all_k.extend([k.strip().lower() for k in str(row).split(";")])
    c = Counter(all_k)
    return pd.DataFrame(c.most_common(n), columns=["Keyword", "Frecuencia"])

def freq_abstract(df, n=25):
    stopwords = {
        "the","a","an","and","or","of","to","in","for","is","are","this","that",
        "with","on","as","by","we","our","be","from","which","at","it","has",
        "have","been","can","also","into","using","based","these","their","they",
        "were","was","its","than","more","not","use","used","such","both","each",
        "paper","study","proposed","approach","model","system","method","results",
        "data","performance","show","work","two","three","new","high","well",
        "used","provides","proposed","present","detect","detection","network",
        "security","learning","machine","deep","intrusion"
    }
    if "Abstract" not in df.columns:
        return pd.DataFrame()
    text = " ".join(df["Abstract"].dropna().astype(str)).lower()
    words = re.findall(r'\b[a-z]{5,}\b', text)
    filtered = [w for w in words if w not in stopwords]
    c = Counter(filtered)
    return pd.DataFrame(c.most_common(n), columns=["Palabra", "Frecuencia"])


# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### 📂 Cargar Dataset")
    source = st.radio("Fuente:", ["📁 Subir CSV local", "🔗 URL de GitHub"])
    df = None

    if source == "📁 Subir CSV local":
        up = st.file_uploader("Sube tu scopus_data.csv", type=["csv"])
        if up:
            df = load_csv(up)
            st.success(f"✅ {len(df)} artículos cargados")
    else:
        url = st.text_input("URL GitHub del CSV:",
            placeholder="https://github.com/usuario/repo/blob/main/scopus_data.csv")
        if url:
            try:
                df = load_github(url)
                st.success(f"✅ {len(df)} artículos desde GitHub")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    if df is not None and "Year" in df.columns:
        st.markdown("---")
        years = sorted(df["Year"].dropna().unique().astype(int))
        rng = st.slider("Filtrar por año:", min(years), max(years), (min(years), max(years)))
        df = df[df["Year"].between(rng[0], rng[1])]

    st.markdown("---")
    st.markdown("**🔬 Pregunta de investigación:**")
    st.info("¿Cómo han evolucionado las técnicas de Machine Learning para la detección de ciberataques en redes (2018–2025)?")
    st.markdown("**🔑 Keywords:**")
    st.code('("machine learning" OR "deep learning")\nAND "intrusion detection"\nAND "network security"\nAND "cybersecurity"')


# ── HEADER ──
st.markdown('<div class="main-title">🔐 ML en Ciberseguridad — Análisis Scopus</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Dashboard bibliométrico interactivo · Ingeniería de Sistemas</div>', unsafe_allow_html=True)

if df is None:
    st.markdown("""
    <div class="info-box">
    <b>👈 Para comenzar:</b> Carga tu archivo <code>scopus_data.csv</code> desde el panel lateral.<br><br>
    El archivo debe ser exportado directamente desde <b>Scopus</b> incluyendo las columnas:<br>
    <code>Authors · Title · Year · Source title · Cited by · Abstract · Author Keywords · DOI</code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── MÉTRICAS ──
st.markdown('<div class="section-header">📊 Resumen General</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📄 Artículos", len(df))
c2.metric("📅 Período", f"{int(df['Year'].min())}–{int(df['Year'].max())}" if "Year" in df.columns else "N/A")
c3.metric("🔖 Total Citas", f"{int(df['Cited by'].sum()):,}" if "Cited by" in df.columns else "N/A")
c4.metric("⭐ Prom. Citas", f"{df['Cited by'].mean():.1f}" if "Cited by" in df.columns else "N/A")
c5.metric("📰 Revistas", df["Source title"].nunique() if "Source title" in df.columns else "N/A")
st.markdown("---")


# ── FILA 1: Publicaciones por año + Revistas ──
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="section-header">📈 Publicaciones por Año</div>', unsafe_allow_html=True)
    pub_year = df.groupby("Year").size().reset_index(name="Publicaciones")
    fig = px.bar(pub_year, x="Year", y="Publicaciones",
                 color="Publicaciones", color_continuous_scale="Purples",
                 text="Publicaciones", labels={"Year":"Año"})
    fig.update_traces(textposition="outside")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      coloraxis_showscale=False, xaxis=dict(dtick=1))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("📌 Crecimiento sostenido del interés en ML+Ciberseguridad en los últimos años.")

with col2:
    st.markdown('<div class="section-header">📰 Por Revista</div>', unsafe_allow_html=True)
    if "Source title" in df.columns:
        journals = df["Source title"].value_counts().reset_index()
        journals.columns = ["Revista", "Artículos"]
        fig2 = px.pie(journals.head(6), names="Revista", values="Artículos", hole=0.4)
        fig2.update_traces(textposition="inside", textinfo="percent+label",
                           textfont_size=10)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                           showlegend=False, margin=dict(t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")


# ── FILA 2: Autores + Artículos citados ──
col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="section-header">👥 Autores más Activos</div>', unsafe_allow_html=True)
    auth_df = top_authors(df, 12)
    if not auth_df.empty:
        fig3 = px.bar(auth_df.sort_values("Publicaciones"), x="Publicaciones", y="Autor",
                      orientation="h", color="Publicaciones", color_continuous_scale="Blues",
                      text="Publicaciones")
        fig3.update_traces(textposition="outside")
        fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           coloraxis_showscale=False, height=420,
                           yaxis=dict(tickfont=dict(size=10)))
        st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.markdown('<div class="section-header">🏆 Artículos más Citados</div>', unsafe_allow_html=True)
    if "Cited by" in df.columns and "Title" in df.columns:
        cited = df[["Title","Year","Cited by"]].sort_values("Cited by", ascending=False).head(8).copy()
        cited["Título"] = cited["Title"].str[:48] + "…"
        fig4 = px.bar(cited.sort_values("Cited by"), x="Cited by", y="Título",
                      orientation="h", color="Cited by", color_continuous_scale="Oranges",
                      text="Cited by", labels={"Cited by":"Citas"})
        fig4.update_traces(textposition="outside")
        fig4.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           coloraxis_showscale=False, height=420,
                           yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")


# ── FILA 3: Keywords treemap + Frecuencia abstracts ──
col5, col6 = st.columns(2)

with col5:
    st.markdown('<div class="section-header">🔑 Keywords de Autores</div>', unsafe_allow_html=True)
    kw_df = top_keywords(df, 20)
    if not kw_df.empty:
        fig5 = px.treemap(kw_df, path=["Keyword"], values="Frecuencia",
                          color="Frecuencia", color_continuous_scale="Teal")
        fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=380)
        st.plotly_chart(fig5, use_container_width=True)
        st.caption("📌 Tamaño proporcional a la frecuencia en los metadatos de autores.")

with col6:
    st.markdown('<div class="section-header">🔤 Términos Frecuentes en Abstracts</div>', unsafe_allow_html=True)
    abs_df = freq_abstract(df, 20)
    if not abs_df.empty:
        fig6 = px.bar(abs_df.sort_values("Frecuencia"), x="Frecuencia", y="Palabra",
                      orientation="h", color="Frecuencia", color_continuous_scale="Reds",
                      text="Frecuencia")
        fig6.update_traces(textposition="outside")
        fig6.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           coloraxis_showscale=False, height=380,
                           yaxis=dict(tickfont=dict(size=10)))
        st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")


# ── TABLA ──
with st.expander("🗂️ Ver tabla completa del dataset"):
    cols = [c for c in ["Title","Authors","Year","Source title","Cited by","Author Keywords","DOI"] if c in df.columns]
    st.dataframe(df[cols].reset_index(drop=True), use_container_width=True)

    csv_export = df[cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV filtrado", csv_export, "scopus_filtrado.csv", "text/csv")


# ── FOOTER ──
st.markdown("""
<hr>
<center style='color:#aaa; font-size:0.82rem;'>
Dashboard desarrollado con Streamlit & Plotly · Datos: Scopus ·
Tema: Machine Learning aplicado a Ciberseguridad
</center>
""", unsafe_allow_html=True)
