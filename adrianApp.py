"""
Banco de Trabajo de Estadística Inferencial
Aplicación Streamlit - Métodos Estadísticos I
INGI02 | FACYT UC
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Estadística Inferencial — INGI02",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# ESTILOS CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* Fondo azul claro */
  .stApp { background-color: #f0f8ff; }
  .stApp > header { background-color: transparent; }

  /* Sidebar */
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #1a3a5c 0%, #1e4d7b 60%, #2563a8 100%);
  }
  [data-testid="stSidebar"] * { color: #e8f4fd !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stRadio label { color: #e8f4fd !important; }

  /* Tarjetas blancas */
  .tarjeta {
      background: white;
      border-radius: 14px;
      padding: 22px 26px;
      margin: 12px 0;
      box-shadow: 0 4px 18px rgba(30,77,123,0.12);
      border-left: 5px solid #2563a8;
  }
  .tarjeta-verde { border-left: 5px solid #16a34a; }
  .tarjeta-roja  { border-left: 5px solid #dc2626; }
  .tarjeta-naranja { border-left: 5px solid #ea580c; }

  /* Títulos */
  h1, h2, h3 { color: #1a3a5c !important; }
  .titulo-principal {
      font-size: 2.2rem;
      font-weight: 800;
      color: #1a3a5c;
      letter-spacing: -1px;
  }
  .subtitulo {
      font-size: 1.1rem;
      color: #2563a8;
      font-weight: 500;
  }

  /* Badge de resultado */
  .resultado-badge {
      background: linear-gradient(135deg, #1e4d7b, #2563a8);
      color: white !important;
      padding: 10px 20px;
      border-radius: 10px;
      font-size: 1.3rem;
      font-weight: 700;
      text-align: center;
      display: inline-block;
      margin: 8px 0;
  }
  .rechaza { background: linear-gradient(135deg, #991b1b, #dc2626) !important; }
  .acepta  { background: linear-gradient(135deg, #14532d, #16a34a) !important; }

  /* Estadísticas descriptivas */
  .stat-box {
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      border-radius: 10px;
      padding: 14px 18px;
      margin: 6px 0;
      text-align: center;
  }
  .stat-valor { font-size: 1.6rem; font-weight: 700; color: #1e40af; }
  .stat-label { font-size: 0.8rem; color: #6b7280; font-weight: 500; text-transform: uppercase; }

  /* Inputs */
  .stTextInput input, .stNumberInput input {
      background: white;
      border: 1.5px solid #bfdbfe;
      border-radius: 8px;
  }
  .stTextInput input:focus, .stNumberInput input:focus {
      border-color: #2563a8;
      box-shadow: 0 0 0 3px rgba(37,99,168,0.15);
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 6px; }
  .stTabs [data-baseweb="tab"] {
      background: white;
      border-radius: 8px 8px 0 0;
      color: #1a3a5c;
      font-weight: 600;
      border: 1.5px solid #bfdbfe;
  }
  .stTabs [aria-selected="true"] {
      background: #2563a8 !important;
      color: white !important;
  }

  /* Expander */
  .streamlit-expanderHeader {
      background: #eff6ff;
      border-radius: 8px;
      font-weight: 600;
      color: #1a3a5c;
  }

  /* Divider */
  hr { border-color: #bfdbfe; }

  /* Métricas */
  [data-testid="stMetricValue"] { color: #1e40af !important; font-weight: 700; }
  [data-testid="stMetricLabel"] { color: #64748b !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────

def parsear_datos(texto: str) -> np.ndarray | None:
    """Convierte cadena separada por comas a array numpy."""
    try:
        valores = [float(x.strip()) for x in texto.replace(";", ",").split(",") if x.strip()]
        return np.array(valores) if valores else None
    except ValueError:
        return None


def estadisticas_descriptivas(datos: np.ndarray) -> dict:
    """Calcula estadísticas descriptivas de una muestra."""
    return {
        "media": np.mean(datos),
        "varianza": np.var(datos, ddof=1),
        "desviacion": np.std(datos, ddof=1),
        "n": len(datos)
    }


def grafica_normal(media=0, desv=1, estadistico=None, valor_critico=None,
                   cola="derecha", titulo="Distribución Normal", color_campana="#2563a8"):
    """Genera gráfica de distribución normal con región de rechazo."""
    x = np.linspace(media - 4*desv, media + 4*desv, 500)
    y = stats.norm.pdf(x, media, desv)

    fig = go.Figure()
    # Campana completa
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                             line=dict(color=color_campana, width=2.5),
                             name="Distribución H₀"))

    # Región de rechazo
    if valor_critico is not None:
        if cola == "derecha":
            x_rej = x[x >= valor_critico]
            y_rej = stats.norm.pdf(x_rej, media, desv)
            x_fill = np.concatenate([[valor_critico], x_rej, [x_rej[-1]]])
            y_fill = np.concatenate([[0], y_rej, [0]])
            fig.add_trace(go.Scatter(x=x_fill, y=y_fill, fill="toself",
                                     fillcolor="rgba(220,38,38,0.35)",
                                     line=dict(color="rgba(0,0,0,0)"), name="Región de Rechazo"))
            fig.add_vline(x=valor_critico, line_dash="dash", line_color="#dc2626",
                          annotation_text=f"Z_c={valor_critico:.3f}", annotation_position="top")
        elif cola == "izquierda":
            x_rej = x[x <= valor_critico]
            y_rej = stats.norm.pdf(x_rej, media, desv)
            x_fill = np.concatenate([[x_rej[0]], x_rej, [valor_critico]])
            y_fill = np.concatenate([[0], y_rej, [0]])
            fig.add_trace(go.Scatter(x=x_fill, y=y_fill, fill="toself",
                                     fillcolor="rgba(220,38,38,0.35)",
                                     line=dict(color="rgba(0,0,0,0)"), name="Región de Rechazo"))
            fig.add_vline(x=valor_critico, line_dash="dash", line_color="#dc2626",
                          annotation_text=f"Z_c={valor_critico:.3f}", annotation_position="top")
        elif cola == "bilateral":
            vc_pos = abs(valor_critico)
            vc_neg = -abs(valor_critico)
            for vc, lado in [(vc_neg, "left"), (vc_pos, "right")]:
                if lado == "left":
                    x_rej = x[x <= vc]
                else:
                    x_rej = x[x >= vc]
                if len(x_rej) == 0:
                    continue
                y_rej = stats.norm.pdf(x_rej, media, desv)
                if lado == "left":
                    xf = np.concatenate([[x_rej[0]], x_rej, [vc]])
                else:
                    xf = np.concatenate([[vc], x_rej, [x_rej[-1]]])
                yf = np.concatenate([[0], y_rej, [0]])
                fig.add_trace(go.Scatter(x=xf, y=yf, fill="toself",
                                         fillcolor="rgba(220,38,38,0.35)",
                                         line=dict(color="rgba(0,0,0,0)"),
                                         name="Región de Rechazo", showlegend=(lado == "left")))
            fig.add_vline(x=vc_neg, line_dash="dash", line_color="#dc2626",
                          annotation_text=f"Z_c={vc_neg:.3f}")
            fig.add_vline(x=vc_pos, line_dash="dash", line_color="#dc2626",
                          annotation_text=f"Z_c={vc_pos:.3f}")

    # Estadístico calculado
    if estadistico is not None:
        fig.add_vline(x=estadistico, line_color="#ea580c", line_width=2.5,
                      annotation_text=f"Z₀={estadistico:.3f}",
                      annotation_position="top right")

    fig.update_layout(
        title=titulo, template="plotly_white",
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        font=dict(family="Georgia, serif", color="#1a3a5c"),
        height=380,
        legend=dict(orientation="h", y=-0.15),
        xaxis_title="Estadístico", yaxis_title="Densidad"
    )
    return fig


def grafica_t(gl, estadistico=None, valor_critico=None, cola="derecha", titulo="Distribución t-Student"):
    """Gráfica distribución t-Student."""
    x = np.linspace(-5, 5, 500)
    y = stats.t.pdf(x, df=gl)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                             line=dict(color="#7c3aed", width=2.5),
                             name=f"t({gl} gl)"))

    if valor_critico is not None:
        if cola == "derecha":
            x_rej = x[x >= valor_critico]
            y_rej = stats.t.pdf(x_rej, df=gl)
            if len(x_rej) > 0:
                fig.add_trace(go.Scatter(
                    x=np.concatenate([[valor_critico], x_rej, [x_rej[-1]]]),
                    y=np.concatenate([[0], y_rej, [0]]),
                    fill="toself", fillcolor="rgba(124,58,237,0.3)",
                    line=dict(color="rgba(0,0,0,0)"), name="Región de Rechazo"))
        fig.add_vline(x=valor_critico, line_dash="dash", line_color="#7c3aed",
                      annotation_text=f"t_c={valor_critico:.3f}", annotation_position="top")

    if estadistico is not None:
        fig.add_vline(x=estadistico, line_color="#ea580c", line_width=2.5,
                      annotation_text=f"t₀={estadistico:.3f}", annotation_position="top right")

    fig.update_layout(
        title=titulo, template="plotly_white",
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        font=dict(family="Georgia, serif", color="#1a3a5c"),
        height=380,
        xaxis_title="Estadístico t", yaxis_title="Densidad"
    )
    return fig


def grafica_potencia(mu0, mu1, sigma, n1, n2, alpha, cola="derecha"):
    """Dos campanas: H0 y H1 para ilustrar potencia."""
    error_std = np.sqrt(sigma**2/n1 + sigma**2/n2)
    x = np.linspace(mu0 - 5*error_std, mu1 + 5*error_std, 600)

    y0 = stats.norm.pdf(x, mu0, error_std)
    y1 = stats.norm.pdf(x, mu1, error_std)

    if cola == "derecha":
        vc = mu0 + stats.norm.ppf(1 - alpha) * error_std
    else:
        vc = mu0 - stats.norm.ppf(1 - alpha) * error_std

    potencia = 1 - stats.norm.cdf(vc, mu1, error_std)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y0, mode="lines",
                             line=dict(color="#2563a8", width=2.5), name="Bajo H₀"))
    fig.add_trace(go.Scatter(x=x, y=y1, mode="lines",
                             line=dict(color="#dc2626", width=2.5), name="Bajo H₁"))

    # Región de potencia (1-β)
    x_pot = x[x >= vc]
    y_pot = stats.norm.pdf(x_pot, mu1, error_std)
    if len(x_pot) > 0:
        fig.add_trace(go.Scatter(
            x=np.concatenate([[vc], x_pot, [x_pot[-1]]]),
            y=np.concatenate([[0], y_pot, [0]]),
            fill="toself", fillcolor="rgba(22,163,74,0.35)",
            line=dict(color="rgba(0,0,0,0)"), name=f"Potencia = {potencia:.4f}"))

    # Región alfa
    x_alpha = x[x >= vc]
    y_alpha = stats.norm.pdf(x_alpha, mu0, error_std)
    if len(x_alpha) > 0:
        fig.add_trace(go.Scatter(
            x=np.concatenate([[vc], x_alpha, [x_alpha[-1]]]),
            y=np.concatenate([[0], y_alpha, [0]]),
            fill="toself", fillcolor="rgba(220,38,38,0.25)",
            line=dict(color="rgba(0,0,0,0)"), name=f"α = {alpha}"))

    fig.add_vline(x=vc, line_dash="dash", line_color="#64748b",
                  annotation_text=f"Vc={vc:.4f}", annotation_position="top")

    fig.update_layout(
        title=f"Potencia de la Prueba (1-β = {potencia:.4f})",
        template="plotly_white",
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        font=dict(family="Georgia, serif", color="#1a3a5c"),
        height=400,
        legend=dict(orientation="h", y=-0.15),
        xaxis_title="X̄₁ - X̄₂", yaxis_title="Densidad"
    )
    return fig, potencia


def grafica_chi2(gl, estadistico=None, valor_critico=None, cola="derecha", titulo="Distribución Chi-Cuadrado"):
    """Gráfica distribución Chi-cuadrado."""
    x = np.linspace(0.01, max(gl*3, 30), 500)
    y = stats.chi2.pdf(x, df=gl)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                             line=dict(color="#0891b2", width=2.5),
                             name=f"χ²({gl} gl)"))

    if valor_critico is not None:
        if cola == "derecha":
            x_rej = x[x >= valor_critico]
            y_rej = stats.chi2.pdf(x_rej, df=gl)
            if len(x_rej) > 0:
                fig.add_trace(go.Scatter(
                    x=np.concatenate([[valor_critico], x_rej, [x_rej[-1]]]),
                    y=np.concatenate([[0], y_rej, [0]]),
                    fill="toself", fillcolor="rgba(220,38,38,0.35)",
                    line=dict(color="rgba(0,0,0,0)"), name="Región de Rechazo"))
        fig.add_vline(x=valor_critico, line_dash="dash", line_color="#dc2626",
                      annotation_text=f"χ²_c={valor_critico:.3f}", annotation_position="top")

    if estadistico is not None:
        fig.add_vline(x=estadistico, line_color="#ea580c", line_width=2.5,
                      annotation_text=f"χ²₀={estadistico:.3f}", annotation_position="top right")

    fig.update_layout(
        title=titulo, template="plotly_white",
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        font=dict(family="Georgia, serif", color="#1a3a5c"),
        height=380,
        xaxis_title="χ²", yaxis_title="Densidad"
    )
    return fig


def grafica_f(gl1, gl2, estadistico=None, valor_critico=None, titulo="Distribución F de Fisher"):
    """Gráfica distribución F."""
    x = np.linspace(0.01, max(gl1, gl2, 10) * 2, 500)
    y = stats.f.pdf(x, dfn=gl1, dfd=gl2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                             line=dict(color="#b45309", width=2.5),
                             name=f"F({gl1},{gl2})"))

    if valor_critico is not None:
        x_rej = x[x >= valor_critico]
        y_rej = stats.f.pdf(x_rej, dfn=gl1, dfd=gl2)
        if len(x_rej) > 0:
            fig.add_trace(go.Scatter(
                x=np.concatenate([[valor_critico], x_rej, [x_rej[-1]]]),
                y=np.concatenate([[0], y_rej, [0]]),
                fill="toself", fillcolor="rgba(180,83,9,0.3)",
                line=dict(color="rgba(0,0,0,0)"), name="Región de Rechazo"))
        fig.add_vline(x=valor_critico, line_dash="dash", line_color="#b45309",
                      annotation_text=f"F_c={valor_critico:.3f}", annotation_position="top")

    if estadistico is not None:
        fig.add_vline(x=estadistico, line_color="#dc2626", line_width=2.5,
                      annotation_text=f"F₀={estadistico:.4f}", annotation_position="top right")

    fig.update_layout(
        title=titulo, template="plotly_white",
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        font=dict(family="Georgia, serif", color="#1a3a5c"),
        height=380,
        xaxis_title="F", yaxis_title="Densidad"
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR — CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <div style='font-size:2.5rem;'>📊</div>
        <div style='font-size:1.1rem; font-weight:800; letter-spacing:1px;'>ESTADÍSTICA</div>
        <div style='font-size:0.85rem; opacity:0.8;'>INFERENCIAL · INGI02</div>
        <div style='font-size:0.75rem; opacity:0.6; margin-top:4px;'>FACYT UC · BDVdigital</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    num_arquitecturas = st.number_input(
        "🏗️ Número de Arquitecturas", min_value=1, max_value=5, value=2, step=1
    )

    st.divider()
    sigma_conocida = st.number_input(
        "σ (desv. estándar poblacional conocida)", min_value=0.0, value=0.18, step=0.01,
        help="Deja en 0 si es desconocida"
    )
    nivel_significancia = st.selectbox(
        "α (nivel de significancia)", [0.01, 0.025, 0.05, 0.10], index=2
    )
    tipo_prueba = st.radio(
        "Tipo de Prueba (H₁)", ["Cola Derecha (>)", "Cola Izquierda (<)", "Bilateral (≠)"]
    )
    diferencia_h0 = st.number_input(
        "Diferencia bajo H₀ (μ₁-μ₂)₀", value=1.2, step=0.1
    )

    st.divider()
    st.markdown("""
    <div style='font-size:0.75rem; opacity:0.7; text-align:center;'>
        Métodos Estadísticos I<br>
        Prueba de Hipótesis · Caja de Herramientas
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ENCABEZADO PRINCIPAL
# ─────────────────────────────────────────────
st.markdown("""
<div class='tarjeta' style='border-left: 5px solid #1a3a5c; background: linear-gradient(135deg, #1a3a5c, #2563a8); color: white;'>
    <div class='titulo-principal' style='color:white !important;'>🔬 Banco de Trabajo Estadístico</div>
    <div style='color:#bfdbfe; margin-top:4px; font-size:1rem;'>
        Estadística Inferencial · Prueba de Hipótesis · INGI02 · FACYT UC
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FASE 1 — INGRESO DE DATOS
# ─────────────────────────────────────────────
st.markdown("## 📥 Fase 1 — Ingreso y Estadísticas Descriptivas")

datos_arquitecturas = {}
estadisticas_arch = {}

nombres_default = ["SwiftChimb (1)", "SwiftPay (2)", "Arquitectura C", "Arquitectura D", "Arquitectura E"]
datos_default = [
    "3.30, 3.42, 3.36, 3.27, 3.45, 3.33, 3.39, 3.30, 3.36, 3.24, 3.40, 3.31, 3.38, 3.35, 3.29",
    "2.05, 2.14, 2.02, 2.08, 2.17, 2.11, 2.05, 2.14, 2.02, 2.08, 2.17, 2.11, 2.09, 1.98, 2.22, 1.95, 2.25, 2.08",
    "", "", ""
]

cols_entrada = st.columns(min(num_arquitecturas, 3))
for i in range(num_arquitecturas):
    col_idx = i % 3
    with cols_entrada[col_idx]:
        st.markdown(f"<div class='tarjeta'>", unsafe_allow_html=True)
        nombre = st.text_input(f"Nombre Arquitectura {i+1}",
                               value=nombres_default[i] if i < len(nombres_default) else f"Arquitectura {i+1}",
                               key=f"nombre_{i}")
        datos_str = st.text_area(
            f"Datos (separados por comas)",
            value=datos_default[i] if i < len(datos_default) else "",
            key=f"datos_{i}", height=100
        )
        datos_arr = parsear_datos(datos_str)
        if datos_arr is not None and len(datos_arr) > 0:
            est = estadisticas_descriptivas(datos_arr)
            datos_arquitecturas[nombre] = datos_arr
            estadisticas_arch[nombre] = est

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"<div class='stat-box'><div class='stat-valor'>{est['media']:.4f}</div><div class='stat-label'>Media X̄</div></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='stat-box'><div class='stat-valor'>{est['varianza']:.6f}</div><div class='stat-label'>Varianza s²</div></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='stat-box'><div class='stat-valor'>{est['desviacion']:.4f}</div><div class='stat-label'>Desv. Est. s</div></div>", unsafe_allow_html=True)
            with c4:
                st.markdown(f"<div class='stat-box'><div class='stat-valor'>{est['n']}</div><div class='stat-label'>n</div></div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Ingresa datos válidos (números separados por comas)")
        st.markdown("</div>", unsafe_allow_html=True)

# Extraer datos de las primeras dos arquitecturas para cálculos
nombres_arq = list(datos_arquitecturas.keys())
if len(nombres_arq) >= 2:
    arq1, arq2 = nombres_arq[0], nombres_arq[1]
    est1, est2 = estadisticas_arch[arq1], estadisticas_arch[arq2]
    x1_bar, s1, n1 = est1["media"], est1["desviacion"], est1["n"]
    x2_bar, s2, n2 = est2["media"], est2["desviacion"], est2["n"]
    s1_2, s2_2 = est1["varianza"], est2["varianza"]
elif len(nombres_arq) == 1:
    arq1 = nombres_arq[0]
    est1 = estadisticas_arch[arq1]
    x1_bar, s1, n1 = est1["media"], est1["desviacion"], est1["n"]
    x2_bar, s2, n2 = 0.0, 1.0, 1
    s1_2, s2_2 = est1["varianza"], 1.0
    arq2 = "N/A"
    est2 = {"media": 0, "varianza": 1, "desviacion": 1, "n": 1}
else:
    x1_bar, s1, n1, s1_2 = 3.343, 0.0589, 15, 0.003467
    x2_bar, s2, n2, s2_2 = 2.095, 0.0788, 18, 0.006215
    arq1, arq2 = "SwiftChimb (1)", "SwiftPay (2)"

# ─────────────────────────────────────────────
# FASE 2 — MENÚ DE FÓRMULAS
# ─────────────────────────────────────────────
st.divider()
st.markdown("## 📐 Fase 2 — Calculadora de Fórmulas Estadísticas")

formulas = [
    "Media (X̄)",
    "Varianza Muestral (s²)",
    "Estadístico Z₀ (dos muestras, σ conocida)",
    "Estadístico t₀ (una muestra)",
    "Estadístico t₀ (dos muestras, varianzas ≠)",
    "Valor p",
    "Error Estándar (dos poblaciones)",
    "Beta (β) y Potencia (1-β)",
    "Tamaño de Muestra (n)",
    "Chi-Cuadrado (χ²)",
    "F de Fisher",
    "Inciso g — Prob. con Valor Crítico Fijo"
]

formula_sel = st.selectbox("📋 Selecciona la Fórmula / Estadístico", formulas)

with st.container():
    st.markdown("<div class='tarjeta'>", unsafe_allow_html=True)

    # ── MEDIA ──
    if formula_sel == formulas[0]:
        st.markdown("### Media Aritmética Muestral")
        st.latex(r"\bar{X} = \frac{\sum_{i=1}^{n} X_i}{n}")
        st.markdown("**Parámetros:**")
        col1, col2 = st.columns(2)
        with col1:
            datos_media_str = st.text_area("Datos (sep. por comas)", value=", ".join([str(round(v,4)) for v in list(datos_arquitecturas.values())[0]]) if datos_arquitecturas else "", key="datos_media")
        datos_media = parsear_datos(datos_media_str)
        if datos_media is not None:
            resultado_media = np.mean(datos_media)
            st.markdown(f"<div class='resultado-badge'>X̄ = {resultado_media:.6f}</div>", unsafe_allow_html=True)

    # ── VARIANZA MUESTRAL ──
    elif formula_sel == formulas[1]:
        st.markdown("### Varianza Muestral")
        st.latex(r"s^2 = \frac{\sum_{i=1}^{n}(X_i - \bar{X})^2}{n-1}")
        datos_var_str = st.text_area("Datos", value=", ".join([str(round(v,4)) for v in list(datos_arquitecturas.values())[0]]) if datos_arquitecturas else "", key="datos_var")
        datos_var = parsear_datos(datos_var_str)
        if datos_var is not None and len(datos_var) > 1:
            res_var = np.var(datos_var, ddof=1)
            res_desv = np.std(datos_var, ddof=1)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div class='resultado-badge'>s² = {res_var:.8f}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='resultado-badge'>s = {res_desv:.6f}</div>", unsafe_allow_html=True)

    # ── Z0 DOS MUESTRAS ──
    elif formula_sel == formulas[2]:
        st.markdown("### Estadístico Z₀ — Diferencia de Medias (σ conocida)")
        st.latex(r"Z_0 = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)_0}{\sqrt{\frac{\sigma^2}{n_1} + \frac{\sigma^2}{n_2}}}")

        cola1, cola2 = st.columns(2)
        with cola1:
            xb1 = st.number_input("X̄₁", value=float(round(x1_bar, 6)), key="z0_xb1", format="%.6f")
            nb1 = st.number_input("n₁", value=int(n1), key="z0_n1", min_value=1)
            sigma_z = st.number_input("σ (poblacional)", value=float(sigma_conocida), key="z0_sigma", min_value=0.001, format="%.4f")
        with cola2:
            xb2 = st.number_input("X̄₂", value=float(round(x2_bar, 6)), key="z0_xb2", format="%.6f")
            nb2 = st.number_input("n₂", value=int(n2), key="z0_n2", min_value=1)
            diff_h0_z = st.number_input("(μ₁-μ₂)₀", value=float(diferencia_h0), key="z0_diff", format="%.4f")

        error_std_z = np.sqrt(sigma_z**2/nb1 + sigma_z**2/nb2)
        z0_val = ((xb1 - xb2) - diff_h0_z) / error_std_z

        cola_prueba_z = "derecha" if "Derecha" in tipo_prueba else ("izquierda" if "Izquierda" in tipo_prueba else "bilateral")
        if cola_prueba_z == "derecha":
            valor_critico_z = stats.norm.ppf(1 - nivel_significancia)
            p_valor_z = 1 - stats.norm.cdf(z0_val)
        elif cola_prueba_z == "izquierda":
            valor_critico_z = stats.norm.ppf(nivel_significancia)
            p_valor_z = stats.norm.cdf(z0_val)
        else:
            valor_critico_z = stats.norm.ppf(1 - nivel_significancia/2)
            p_valor_z = 2 * (1 - stats.norm.cdf(abs(z0_val)))

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.markdown(f"<div class='resultado-badge'>Z₀ = {z0_val:.4f}</div>", unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"<div class='resultado-badge'>Valor p = {p_valor_z:.6f}</div>", unsafe_allow_html=True)
        with col_r3:
            decision = "RECHAZA H₀" if p_valor_z < nivel_significancia else "NO Rechaza H₀"
            css_clase = "rechaza" if p_valor_z < nivel_significancia else "acepta"
            st.markdown(f"<div class='resultado-badge {css_clase}'>{decision}</div>", unsafe_allow_html=True)

        st.markdown(f"**Error Estándar:** √(σ²/n₁ + σ²/n₂) = {error_std_z:.6f}")
        st.markdown(f"**Z crítico ({cola_prueba_z}):** {valor_critico_z:.4f}")

    # ── T0 UNA MUESTRA ──
    elif formula_sel == formulas[3]:
        st.markdown("### Estadístico t₀ — Una Muestra")
        st.latex(r"t_0 = \frac{\bar{X} - \mu_0}{s / \sqrt{n}}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            xb_t1 = st.number_input("X̄", value=float(round(x1_bar, 6)), key="t1_xb", format="%.6f")
        with col2:
            mu0_t1 = st.number_input("μ₀ (hipótesis)", value=float(diferencia_h0), key="t1_mu0", format="%.4f")
        with col3:
            s_t1 = st.number_input("s (desv. muestral)", value=float(round(s1, 6)), key="t1_s", min_value=0.0001, format="%.6f")
        with col4:
            n_t1 = st.number_input("n", value=int(n1), key="t1_n", min_value=2)

        t0_val = (xb_t1 - mu0_t1) / (s_t1 / np.sqrt(n_t1))
        gl_t1 = n_t1 - 1
        p_t1 = 1 - stats.t.cdf(t0_val, df=gl_t1)

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.markdown(f"<div class='resultado-badge'>t₀ = {t0_val:.4f}</div>", unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"<div class='resultado-badge'>gl = {gl_t1}</div>", unsafe_allow_html=True)
        with col_r3:
            st.markdown(f"<div class='resultado-badge'>Valor p ≈ {p_t1:.4f}</div>", unsafe_allow_html=True)

    # ── T0 DOS MUESTRAS ──
    elif formula_sel == formulas[4]:
        st.markdown("### Estadístico t₀ — Dos Muestras (Varianzas Desconocidas ≠)")
        st.latex(r"t_0 = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1-\mu_2)_0}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}")
        st.latex(r"\nu = \frac{\left(\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1}+\frac{(s_2^2/n_2)^2}{n_2-1}}")

        col1, col2 = st.columns(2)
        with col1:
            xb1_t2 = st.number_input("X̄₁", value=float(round(x1_bar, 6)), key="t2_xb1", format="%.6f")
            s1_t2  = st.number_input("s₁", value=float(round(s1, 6)), key="t2_s1", min_value=0.0001, format="%.6f")
            n1_t2  = st.number_input("n₁", value=int(n1), key="t2_n1", min_value=2)
        with col2:
            xb2_t2 = st.number_input("X̄₂", value=float(round(x2_bar, 6)), key="t2_xb2", format="%.6f")
            s2_t2  = st.number_input("s₂", value=float(round(s2, 6)), key="t2_s2", min_value=0.0001, format="%.6f")
            n2_t2  = st.number_input("n₂", value=int(n2), key="t2_n2", min_value=2)
        diff_h0_t2 = st.number_input("(μ₁-μ₂)₀", value=float(diferencia_h0), key="t2_diff", format="%.4f")

        s1_2_t2 = s1_t2**2
        s2_2_t2 = s2_t2**2
        numerador_t2 = (xb1_t2 - xb2_t2) - diff_h0_t2
        denom_t2 = np.sqrt(s1_2_t2/n1_t2 + s2_2_t2/n2_t2)
        t0_t2 = numerador_t2 / denom_t2

        # Grados de libertad Welch-Satterthwaite
        num_gl = (s1_2_t2/n1_t2 + s2_2_t2/n2_t2)**2
        den_gl = (s1_2_t2/n1_t2)**2/(n1_t2-1) + (s2_2_t2/n2_t2)**2/(n2_t2-1)
        gl_t2 = int(num_gl / den_gl)

        cola_prueba_t2 = "derecha" if "Derecha" in tipo_prueba else ("izquierda" if "Izquierda" in tipo_prueba else "bilateral")
        if cola_prueba_t2 == "derecha":
            p_t2 = 1 - stats.t.cdf(t0_t2, df=gl_t2)
            tc_t2 = stats.t.ppf(1 - nivel_significancia, df=gl_t2)
        else:
            p_t2 = stats.t.cdf(t0_t2, df=gl_t2)
            tc_t2 = stats.t.ppf(nivel_significancia, df=gl_t2)

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.markdown(f"<div class='resultado-badge'>t₀ = {t0_t2:.4f}</div>", unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"<div class='resultado-badge'>gl = {gl_t2}</div>", unsafe_allow_html=True)
        with col_r3:
            st.markdown(f"<div class='resultado-badge'>Valor p = {p_t2:.4f}</div>", unsafe_allow_html=True)
        with col_r4:
            dec = "RECHAZA H₀" if p_t2 < nivel_significancia else "NO Rechaza H₀"
            css_c = "rechaza" if p_t2 < nivel_significancia else "acepta"
            st.markdown(f"<div class='resultado-badge {css_c}'>{dec}</div>", unsafe_allow_html=True)
        st.markdown(f"**t crítico:** {tc_t2:.4f} | **Error estándar:** {denom_t2:.6f}")

    # ── VALOR P ──
    elif formula_sel == formulas[5]:
        st.markdown("### Valor p")
        st.latex(r"\text{Valor } p = P(Z > Z_0 \mid H_0 \text{ verdadera})")
        st.info("El **valor p** es la probabilidad de obtener un estadístico tan extremo o más que el observado, asumiendo que H₀ es verdadera. Si **valor p < α**, se rechaza H₀.")

        col1, col2, col3 = st.columns(3)
        with col1:
            estadistico_vp = st.number_input("Estadístico calculado (Z₀ o t₀)", value=0.763, key="vp_est", format="%.4f")
        with col2:
            tipo_dist = st.selectbox("Distribución", ["Normal (Z)", "t-Student"], key="vp_dist")
        with col3:
            if tipo_dist == "t-Student":
                gl_vp = st.number_input("Grados de libertad", value=31, key="vp_gl", min_value=1)

        cola_vp = "derecha" if "Derecha" in tipo_prueba else ("izquierda" if "Izquierda" in tipo_prueba else "bilateral")

        if tipo_dist == "Normal (Z)":
            if cola_vp == "derecha":
                p_vp = 1 - stats.norm.cdf(estadistico_vp)
            elif cola_vp == "izquierda":
                p_vp = stats.norm.cdf(estadistico_vp)
            else:
                p_vp = 2 * (1 - stats.norm.cdf(abs(estadistico_vp)))
        else:
            if cola_vp == "derecha":
                p_vp = 1 - stats.t.cdf(estadistico_vp, df=gl_vp)
            elif cola_vp == "izquierda":
                p_vp = stats.t.cdf(estadistico_vp, df=gl_vp)
            else:
                p_vp = 2 * (1 - stats.t.cdf(abs(estadistico_vp), df=gl_vp))

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown(f"<div class='resultado-badge'>Valor p = {p_vp:.6f}</div>", unsafe_allow_html=True)
        with col_r2:
            dec_vp = "RECHAZA H₀" if p_vp < nivel_significancia else "NO Rechaza H₀"
            css_vp = "rechaza" if p_vp < nivel_significancia else "acepta"
            st.markdown(f"<div class='resultado-badge {css_vp}'>{dec_vp}</div>", unsafe_allow_html=True)

    # ── ERROR ESTÁNDAR ──
    elif formula_sel == formulas[6]:
        st.markdown("### Error Estándar — Dos Poblaciones")
        st.latex(r"E = \sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}")

        col1, col2 = st.columns(2)
        with col1:
            sig1_ee = st.number_input("σ₁² (varianza poblacional 1)", value=float(sigma_conocida**2), key="ee_s1", min_value=0.0, format="%.6f")
            n1_ee = st.number_input("n₁", value=int(n1), key="ee_n1", min_value=1)
        with col2:
            sig2_ee = st.number_input("σ₂² (varianza poblacional 2)", value=float(sigma_conocida**2), key="ee_s2", min_value=0.0, format="%.6f")
            n2_ee = st.number_input("n₂", value=int(n2), key="ee_n2", min_value=1)

        ee_val = np.sqrt(sig1_ee/n1_ee + sig2_ee/n2_ee)
        st.markdown(f"<div class='resultado-badge'>E = {ee_val:.6f}</div>", unsafe_allow_html=True)

    # ── BETA Y POTENCIA ──
    elif formula_sel == formulas[7]:
        st.markdown("### Beta (β) y Potencia (1-β)")
        st.latex(r"\beta = P\left(\bar{X}_1 - \bar{X}_2 < V_c^* \mid H_1\right) = P\left(Z < \frac{V_c^* - \mu_1}{\sigma/\sqrt{n}}\right)")
        st.latex(r"1 - \beta = P\left(Z > \frac{V_c^* - (\mu_1-\mu_2)}{\sqrt{\sigma_1^2/n_1+\sigma_2^2/n_2}}\right)")

        col1, col2 = st.columns(2)
        with col1:
            mu0_pot = st.number_input("(μ₁-μ₂)₀ bajo H₀", value=float(diferencia_h0), key="pot_mu0", format="%.4f")
            mu1_pot = st.number_input("(μ₁-μ₂) verdadera bajo H₁", value=float(diferencia_h0)+0.15, key="pot_mu1", format="%.4f")
            sigma_pot = st.number_input("σ (poblacional)", value=float(sigma_conocida), key="pot_sigma", min_value=0.001, format="%.4f")
        with col2:
            n1_pot = st.number_input("n₁", value=int(n1), key="pot_n1", min_value=1)
            n2_pot = st.number_input("n₂", value=int(n2), key="pot_n2", min_value=1)

        ee_pot = np.sqrt(sigma_pot**2/n1_pot + sigma_pot**2/n2_pot)
        vc_pot = mu0_pot + stats.norm.ppf(1 - nivel_significancia) * ee_pot
        beta_val = stats.norm.cdf(vc_pot, mu1_pot, ee_pot)
        potencia_val = 1 - beta_val

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.markdown(f"<div class='resultado-badge'>Vc* = {vc_pot:.4f}</div>", unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"<div class='resultado-badge'>β = {beta_val:.4f}</div>", unsafe_allow_html=True)
        with col_r3:
            st.markdown(f"<div class='resultado-badge acepta'>1-β = {potencia_val:.4f}</div>", unsafe_allow_html=True)

    # ── TAMAÑO DE MUESTRA ──
    elif formula_sel == formulas[8]:
        st.markdown("### Tamaño de Muestra (n)")
        st.latex(r"n = \left(\frac{(Z_{1-\alpha} + Z_{1-\beta})\sigma}{\Delta}\right)^2")
        st.markdown("Donde Δ = diferencia real a detectar (μ₁ - μ₂)₁ − (μ₁ − μ₂)₀")

        col1, col2 = st.columns(2)
        with col1:
            alpha_n = st.number_input("α", value=float(nivel_significancia), key="n_alpha", min_value=0.001, max_value=0.5, format="%.3f")
            beta_n = st.number_input("β (1-potencia)", value=0.20, key="n_beta", min_value=0.001, max_value=0.5, format="%.3f")
        with col2:
            sigma_n = st.number_input("σ (poblacional)", value=float(sigma_conocida), key="n_sigma", min_value=0.001, format="%.4f")
            delta_n = st.number_input("Δ (diferencia a detectar)", value=0.10, key="n_delta", min_value=0.001, format="%.4f")

        z_alpha_n = stats.norm.ppf(1 - alpha_n)
        z_beta_n = stats.norm.ppf(1 - beta_n)
        n_calc = ((z_alpha_n + z_beta_n) * sigma_n / delta_n) ** 2
        n_ceil = int(np.ceil(n_calc))

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.markdown(f"<div class='resultado-badge'>Z_α = {z_alpha_n:.4f}</div>", unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"<div class='resultado-badge'>Z_β = {z_beta_n:.4f}</div>", unsafe_allow_html=True)
        with col_r3:
            st.markdown(f"<div class='resultado-badge acepta'>n = {n_ceil} observaciones</div>", unsafe_allow_html=True)

    # ── CHI-CUADRADO ──
    elif formula_sel == formulas[9]:
        st.markdown("### Chi-Cuadrado (χ²)")
        st.latex(r"\chi^2 = \frac{s^2(n-1)}{\sigma_0^2}")

        col1, col2, col3 = st.columns(3)
        with col1:
            s2_chi = st.number_input("s² (varianza muestral)", value=float(round(s2_2, 8)), key="chi_s2", min_value=0.0, format="%.8f")
        with col2:
            n_chi = st.number_input("n", value=int(n2), key="chi_n", min_value=2)
        with col3:
            sigma0_chi = st.number_input("σ₀² (varianza bajo H₀)", value=0.005, key="chi_sigma0", min_value=0.0001, format="%.6f")

        chi2_val = s2_chi * (n_chi - 1) / sigma0_chi
        gl_chi = n_chi - 1
        p_chi = 1 - stats.chi2.cdf(chi2_val, df=gl_chi)
        vc_chi = stats.chi2.ppf(1 - nivel_significancia, df=gl_chi)

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.markdown(f"<div class='resultado-badge'>χ²₀ = {chi2_val:.4f}</div>", unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"<div class='resultado-badge'>gl = {gl_chi}</div>", unsafe_allow_html=True)
        with col_r3:
            st.markdown(f"<div class='resultado-badge'>Valor p = {p_chi:.4f}</div>", unsafe_allow_html=True)
        with col_r4:
            dec_chi = "RECHAZA H₀" if p_chi < nivel_significancia else "NO Rechaza H₀"
            css_chi = "rechaza" if p_chi < nivel_significancia else "acepta"
            st.markdown(f"<div class='resultado-badge {css_chi}'>{dec_chi}</div>", unsafe_allow_html=True)

    # ── F DE FISHER ──
    elif formula_sel == formulas[10]:
        st.markdown("### F de Fisher — Razón de Varianzas")
        st.latex(r"F_0 = \frac{s_1^2}{s_2^2}")

        col1, col2 = st.columns(2)
        with col1:
            s1_f = st.number_input("s₁² (varianza muestral 1)", value=float(round(s1_2, 8)), key="f_s1", min_value=0.0001, format="%.8f")
            n1_f = st.number_input("n₁", value=int(n1), key="f_n1", min_value=2)
        with col2:
            s2_f = st.number_input("s₂² (varianza muestral 2)", value=float(round(s2_2, 8)), key="f_s2", min_value=0.0001, format="%.8f")
            n2_f = st.number_input("n₂", value=int(n2), key="f_n2", min_value=2)

        f0_val = s1_f / s2_f
        gl1_f = n1_f - 1
        gl2_f = n2_f - 1
        p_f = 1 - stats.f.cdf(f0_val, dfn=gl1_f, dfd=gl2_f)
        vc_f = stats.f.ppf(1 - nivel_significancia, dfn=gl1_f, dfd=gl2_f)

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.markdown(f"<div class='resultado-badge'>F₀ = {f0_val:.4f}</div>", unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"<div class='resultado-badge'>gl: ({gl1_f},{gl2_f})</div>", unsafe_allow_html=True)
        with col_r3:
            st.markdown(f"<div class='resultado-badge'>Valor p = {p_f:.4f}</div>", unsafe_allow_html=True)
        with col_r4:
            dec_f = "RECHAZA H₀" if p_f < nivel_significancia else "NO Rechaza H₀"
            css_f = "rechaza" if p_f < nivel_significancia else "acepta"
            st.markdown(f"<div class='resultado-badge {css_f}'>{dec_f}</div>", unsafe_allow_html=True)

    # ── INCISO G ──
    elif formula_sel == formulas[11]:
        st.markdown("### Inciso g — Probabilidad con Valor Crítico Fijo (Vc)")
        st.latex(r"P(\text{Liberar}) = P\left(\bar{X}_1 - \bar{X}_2 > V_c \mid H_0\right) = P\left(Z > \frac{V_c - (\mu_1-\mu_2)_0}{\sqrt{\sigma_1^2/n_1+\sigma_2^2/n_2}}\right)")

        col1, col2, col3 = st.columns(3)
        with col1:
            vc_g = st.number_input("Valor Crítico fijo (Vc)", value=1.30, key="g_vc", format="%.4f")
        with col2:
            sigma_g = st.number_input("σ (poblacional)", value=float(sigma_conocida), key="g_sigma", min_value=0.001, format="%.4f")
        with col3:
            diff_h0_g = st.number_input("(μ₁-μ₂)₀", value=float(diferencia_h0), key="g_diff", format="%.4f")

        n1_g = st.number_input("n₁", value=int(n1), key="g_n1", min_value=1)
        n2_g = st.number_input("n₂", value=int(n2), key="g_n2", min_value=1)

        ee_g = np.sqrt(sigma_g**2/n1_g + sigma_g**2/n2_g)
        z_g = (vc_g - diff_h0_g) / ee_g
        prob_liberar = 1 - stats.norm.cdf(z_g)

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.markdown(f"<div class='resultado-badge'>Z = {z_g:.4f}</div>", unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"<div class='resultado-badge'>Error Estándar = {ee_g:.4f}</div>", unsafe_allow_html=True)
        with col_r3:
            st.markdown(f"<div class='resultado-badge acepta'>P(Liberar) = {prob_liberar:.4f}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FASE 3 — PANEL DE VISUALIZACIONES (7 INCISOS)
# ─────────────────────────────────────────────
st.divider()
st.markdown("## 📈 Fase 3 — Panel de Visualizaciones (Los 7 Incisos)")
st.markdown("*Basado en el examen: SwiftChimb (1) vs SwiftPay (2) — FACYT UC*")

# Calcular estadísticos globales para gráficas
sigma_graf = sigma_conocida if sigma_conocida > 0 else 0.18
ee_graf = np.sqrt(sigma_graf**2/n1 + sigma_graf**2/n2)
z0_graf = ((x1_bar - x2_bar) - diferencia_h0) / ee_graf

cola_graf = "derecha" if "Derecha" in tipo_prueba else ("izquierda" if "Izquierda" in tipo_prueba else "bilateral")
if cola_graf == "derecha":
    vc_graf = stats.norm.ppf(1 - nivel_significancia)
    p_val_graf = 1 - stats.norm.cdf(z0_graf)
elif cola_graf == "izquierda":
    vc_graf = stats.norm.ppf(nivel_significancia)
    p_val_graf = stats.norm.cdf(z0_graf)
else:
    vc_graf = stats.norm.ppf(1 - nivel_significancia/2)
    p_val_graf = 2 * (1 - stats.norm.cdf(abs(z0_graf)))

# t-Student con varianzas desiguales
s1_2_g = s1**2
s2_2_g = s2**2
denom_t_g = np.sqrt(s1_2_g/n1 + s2_2_g/n2)
t0_g = ((x1_bar - x2_bar) - diferencia_h0) / denom_t_g
num_gl_g = (s1_2_g/n1 + s2_2_g/n2)**2
den_gl_g = (s1_2_g/n1)**2/(n1-1) + (s2_2_g/n2)**2/(n2-1)
gl_g = max(int(num_gl_g / den_gl_g), 1)
tc_g = stats.t.ppf(1 - nivel_significancia, df=gl_g)
p_t_g = 1 - stats.t.cdf(t0_g, df=gl_g)

tab_a, tab_b, tab_c, tab_d, tab_e, tab_f, tab_g = st.tabs([
    "a) Prueba Base (Z)",
    "b) Varianzas Desconocidas (t)",
    "c) Potencia",
    "d) Tamaño de Muestra",
    "e) Una Varianza (χ²)",
    "f) Razón de Varianzas (F)",
    "g) Valor Crítico Fijo"
])

# ── INCISO A ──
with tab_a:
    st.markdown("<div class='tarjeta'>", unsafe_allow_html=True)
    st.markdown(f"""
    **Inciso a) — Prueba Base con σ conocida**
    - H₀: (μ₁-μ₂) = {diferencia_h0} | H₁: (μ₁-μ₂) > {diferencia_h0}
    - X̄₁ = {x1_bar:.4f} | X̄₂ = {x2_bar:.4f} | σ = {sigma_graf} | n₁={n1}, n₂={n2}
    - Error Estándar = {ee_graf:.4f} | **Z₀ = {z0_graf:.4f}** | Z_crítico = {vc_graf:.4f} | **Valor p = {p_val_graf:.4f}**
    """)
    dec_a = "✅ SE RECHAZA H₀" if p_val_graf < nivel_significancia else "❌ NO se rechaza H₀"
    css_a = "rechaza" if p_val_graf < nivel_significancia else "acepta"
    st.markdown(f"<div class='resultado-badge {css_a}'>{dec_a}</div>", unsafe_allow_html=True)
    fig_a = grafica_normal(0, 1, z0_graf, vc_graf, cola_graf,
                           f"Inciso a) — Distribución Normal bajo H₀ | Z₀={z0_graf:.4f}")
    st.plotly_chart(fig_a, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── INCISO B ──
with tab_b:
    st.markdown("<div class='tarjeta'>", unsafe_allow_html=True)
    st.markdown(f"""
    **Inciso b) — Varianzas Desconocidas (t-Student, Welch-Satterthwaite)**
    - s₁² = {s1_2_g:.6f} | s₂² = {s2_2_g:.6f}
    - Grados de libertad (ν) = {gl_g} | **t₀ = {t0_g:.4f}** | t_crítico = {tc_g:.4f} | **Valor p = {p_t_g:.4f}**
    """)
    dec_b = "✅ SE RECHAZA H₀" if p_t_g < nivel_significancia else "❌ NO se rechaza H₀"
    css_b = "rechaza" if p_t_g < nivel_significancia else "acepta"
    st.markdown(f"<div class='resultado-badge {css_b}'>{dec_b}</div>", unsafe_allow_html=True)
    fig_b = grafica_t(gl_g, t0_g, tc_g, "derecha",
                      f"Inciso b) — t-Student ({gl_g} gl) | t₀={t0_g:.4f}")
    st.plotly_chart(fig_b, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── INCISO C ──
with tab_c:
    st.markdown("<div class='tarjeta'>", unsafe_allow_html=True)
    mu0_c = diferencia_h0
    mu1_c = diferencia_h0 + 0.15  # diferencia verdadera
    st.markdown(f"**Inciso c) — Potencia de la Prueba**")
    col1c, col2c = st.columns(2)
    with col1c:
        mu1_c_inp = st.number_input("Diferencia verdadera (μ₁-μ₂) bajo H₁",
                                    value=mu0_c + 0.15, key="c_mu1", format="%.4f")
    fig_c, pot_c = grafica_potencia(mu0_c, mu1_c_inp, sigma_graf, n1, n2, nivel_significancia)
    st.markdown(f"**Potencia (1-β) = {pot_c:.4f}** | β = {1-pot_c:.4f}")
    st.plotly_chart(fig_c, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── INCISO D ──
with tab_d:
    st.markdown("<div class='tarjeta'>", unsafe_allow_html=True)
    st.markdown("**Inciso d) — Tamaño de Muestra para Potencia Deseada**")
    col1d, col2d = st.columns(2)
    with col1d:
        beta_d = st.number_input("β deseada", value=0.20, key="d_beta", min_value=0.01, max_value=0.49, format="%.2f")
        delta_d = st.number_input("Δ (diferencia a detectar)", value=0.10, key="d_delta", min_value=0.001, format="%.4f")

    z_alpha_d = stats.norm.ppf(1 - nivel_significancia)
    z_beta_d = stats.norm.ppf(1 - beta_d)
    n_d = ((z_alpha_d + z_beta_d) * sigma_graf / delta_d) ** 2
    n_d_ceil = int(np.ceil(n_d))

    st.markdown(f"""
    - Z_α = {z_alpha_d:.4f} | Z_β = {z_beta_d:.4f}
    - **n requerido = {n_d_ceil} observaciones por arquitectura**
    """)
    st.markdown(f"<div class='resultado-badge acepta'>n = {n_d_ceil} obs/arquitectura</div>", unsafe_allow_html=True)

    # Gráfica de n vs Δ
    deltas = np.linspace(0.01, 0.5, 200)
    ns = np.ceil(((z_alpha_d + z_beta_d) * sigma_graf / deltas) ** 2)
    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(x=deltas, y=ns, mode="lines",
                               line=dict(color="#2563a8", width=2.5)))
    fig_d.add_vline(x=delta_d, line_dash="dash", line_color="#dc2626",
                    annotation_text=f"Δ={delta_d}", annotation_position="top")
    fig_d.add_hline(y=n_d_ceil, line_dash="dash", line_color="#16a34a",
                    annotation_text=f"n={n_d_ceil}", annotation_position="right")
    fig_d.update_layout(
        title=f"Inciso d) — Tamaño de Muestra vs Diferencia a Detectar (α={nivel_significancia}, β={beta_d})",
        xaxis_title="Δ (diferencia)", yaxis_title="n requerido",
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="#f8fafc",
        font=dict(family="Georgia, serif", color="#1a3a5c"), height=380
    )
    st.plotly_chart(fig_d, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── INCISO E ──
with tab_e:
    st.markdown("<div class='tarjeta'>", unsafe_allow_html=True)
    st.markdown("**Inciso e) — Una Varianza (Chi-Cuadrado)**")
    st.markdown("*Se sospecha que la varianza de SwiftPay (2) es mayor a 0.005 s²*")

    col1e, col2e = st.columns(2)
    with col1e:
        s2_e = st.number_input("s² muestral", value=float(round(s2_2, 8)), key="e_s2", min_value=0.0, format="%.8f")
        n_e = st.number_input("n", value=int(n2), key="e_n", min_value=2)
    with col2e:
        sigma0_e = st.number_input("σ₀² bajo H₀", value=0.005, key="e_sigma0", min_value=0.0001, format="%.6f")

    chi2_e = s2_e * (n_e - 1) / sigma0_e
    gl_e = n_e - 1
    vc_chi_e = stats.chi2.ppf(1 - nivel_significancia, df=gl_e)
    p_chi_e = 1 - stats.chi2.cdf(chi2_e, df=gl_e)

    col_re1, col_re2, col_re3 = st.columns(3)
    with col_re1:
        st.markdown(f"<div class='resultado-badge'>χ²₀ = {chi2_e:.4f}</div>", unsafe_allow_html=True)
    with col_re2:
        st.markdown(f"<div class='resultado-badge'>χ²_crítico = {vc_chi_e:.4f}</div>", unsafe_allow_html=True)
    with col_re3:
        dec_e = "✅ RECHAZA H₀" if p_chi_e < nivel_significancia else "❌ NO Rechaza H₀"
        css_e = "rechaza" if p_chi_e < nivel_significancia else "acepta"
        st.markdown(f"<div class='resultado-badge {css_e}'>{dec_e}</div>", unsafe_allow_html=True)
    st.markdown(f"**Valor p = {p_chi_e:.4f}** | gl = {gl_e}")

    fig_e = grafica_chi2(gl_e, chi2_e, vc_chi_e, "derecha",
                         f"Inciso e) — χ²({gl_e} gl) | χ²₀={chi2_e:.4f}")
    st.plotly_chart(fig_e, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── INCISO F ──
with tab_f:
    st.markdown("<div class='tarjeta'>", unsafe_allow_html=True)
    st.markdown("**Inciso f) — Razón de Varianzas (F de Fisher)**")
    st.markdown("*¿La varianza de SwiftPay (2) es al menos un 10% mayor que SwiftChimb (1)?*")

    col1f, col2f = st.columns(2)
    with col1f:
        s1_f2 = st.number_input("s₁² (Arq. 1)", value=float(round(s1_2, 8)), key="f2_s1", min_value=0.0001, format="%.8f")
        n1_f2 = st.number_input("n₁", value=int(n1), key="f2_n1", min_value=2)
    with col2f:
        s2_f2 = st.number_input("s₂² (Arq. 2)", value=float(round(s2_2, 8)), key="f2_s2", min_value=0.0001, format="%.8f")
        n2_f2 = st.number_input("n₂", value=int(n2), key="f2_n2", min_value=2)
    factor_f = st.number_input("Factor mínimo (ej. 1.10 para 10% mayor)", value=1.10, key="f2_factor", format="%.2f")

    f0_f2 = s2_f2 / (factor_f * s1_f2)
    gl1_f2 = n2_f2 - 1
    gl2_f2 = n1_f2 - 1
    vc_f2 = stats.f.ppf(1 - nivel_significancia, dfn=gl1_f2, dfd=gl2_f2)
    p_f2 = 1 - stats.f.cdf(f0_f2, dfn=gl1_f2, dfd=gl2_f2)

    col_rf1, col_rf2, col_rf3 = st.columns(3)
    with col_rf1:
        st.markdown(f"<div class='resultado-badge'>F₀ = {f0_f2:.4f}</div>", unsafe_allow_html=True)
    with col_rf2:
        st.markdown(f"<div class='resultado-badge'>F_crítico = {vc_f2:.4f}</div>", unsafe_allow_html=True)
    with col_rf3:
        dec_f2 = "✅ RECHAZA H₀" if p_f2 < nivel_significancia else "❌ NO Rechaza H₀"
        css_f2 = "rechaza" if p_f2 < nivel_significancia else "acepta"
        st.markdown(f"<div class='resultado-badge {css_f2}'>{dec_f2}</div>", unsafe_allow_html=True)
    st.markdown(f"**Valor p = {p_f2:.4f}** | gl: ({gl1_f2}, {gl2_f2})")

    fig_f2 = grafica_f(gl1_f2, gl2_f2, f0_f2, vc_f2,
                       f"Inciso f) — F({gl1_f2},{gl2_f2}) | F₀={f0_f2:.4f}")
    st.plotly_chart(fig_f2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── INCISO G ──
with tab_g:
    st.markdown("<div class='tarjeta'>", unsafe_allow_html=True)
    st.markdown("**Inciso g) — Valor Crítico Fijo en el Eje X**")

    col1g, col2g = st.columns(2)
    with col1g:
        vc_g_tab = st.number_input("Valor Crítico fijo (Vc) en segundos", value=1.30, key="g_vc_tab", format="%.4f")
        sigma_g_tab = st.number_input("σ (poblacional)", value=float(sigma_graf), key="g_sigma_tab", min_value=0.001, format="%.4f")
    with col2g:
        diff_h0_g_tab = st.number_input("(μ₁-μ₂)₀", value=float(diferencia_h0), key="g_diff_tab", format="%.4f")
        n1_g_tab = st.number_input("n₁", value=int(n1), key="g_n1_tab", min_value=1)
        n2_g_tab = st.number_input("n₂", value=int(n2), key="g_n2_tab", min_value=1)

    ee_g_tab = np.sqrt(sigma_g_tab**2/n1_g_tab + sigma_g_tab**2/n2_g_tab)
    z_g_tab = (vc_g_tab - diff_h0_g_tab) / ee_g_tab
    prob_lib = 1 - stats.norm.cdf(z_g_tab)

    col_rg1, col_rg2 = st.columns(2)
    with col_rg1:
        st.markdown(f"<div class='resultado-badge'>Z correspondiente = {z_g_tab:.4f}</div>", unsafe_allow_html=True)
    with col_rg2:
        st.markdown(f"<div class='resultado-badge acepta'>P(Liberar despliegue) = {prob_lib:.4f}</div>", unsafe_allow_html=True)

    # Gráfica en escala de X̄₁-X̄₂ original
    mu_g = diff_h0_g_tab
    x_g = np.linspace(mu_g - 5*ee_g_tab, mu_g + 5*ee_g_tab, 500)
    y_g = stats.norm.pdf(x_g, mu_g, ee_g_tab)

    fig_g = go.Figure()
    fig_g.add_trace(go.Scatter(x=x_g, y=y_g, mode="lines",
                               line=dict(color="#2563a8", width=2.5), name="Distribución bajo H₀"))

    # Región de liberar (derecha del Vc)
    x_lib = x_g[x_g >= vc_g_tab]
    y_lib = stats.norm.pdf(x_lib, mu_g, ee_g_tab)
    if len(x_lib) > 0:
        fig_g.add_trace(go.Scatter(
            x=np.concatenate([[vc_g_tab], x_lib, [x_lib[-1]]]),
            y=np.concatenate([[0], y_lib, [0]]),
            fill="toself", fillcolor="rgba(22,163,74,0.4)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"P(Liberar) = {prob_lib:.4f}"))

    fig_g.add_vline(x=vc_g_tab, line_dash="dash", line_color="#dc2626", line_width=2,
                    annotation_text=f"Vc = {vc_g_tab}", annotation_position="top")
    fig_g.add_vline(x=x1_bar - x2_bar, line_color="#ea580c", line_width=2,
                    annotation_text=f"X̄₁-X̄₂ = {x1_bar-x2_bar:.4f}", annotation_position="top right")

    fig_g.update_layout(
        title=f"Inciso g) — Nuevo Valor Crítico Vc={vc_g_tab} | P(Liberar)={prob_lib:.4f}",
        xaxis_title="X̄₁ - X̄₂ (segundos)", yaxis_title="Densidad",
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="#f8fafc",
        font=dict(family="Georgia, serif", color="#1a3a5c"), height=400,
        legend=dict(orientation="h", y=-0.15)
    )
    st.plotly_chart(fig_g, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RESUMEN EJECUTIVO
# ─────────────────────────────────────────────
st.divider()
st.markdown("## 📋 Resumen Ejecutivo del Análisis")

resumen_data = {
    "Inciso": ["a) Prueba Z (σ conocida)", "b) Prueba t (var. desconocidas)", "e) Chi-Cuadrado", "f) F de Fisher"],
    "Estadístico": [f"Z₀ = {z0_graf:.4f}", f"t₀ = {t0_g:.4f}", "Ver Inciso e)", "Ver Inciso f)"],
    "Valor crítico": [f"Z_c = {vc_graf:.4f}", f"t_c = {tc_g:.4f}", "—", "—"],
    "Valor p": [f"{p_val_graf:.4f}", f"{p_t_g:.4f}", "—", "—"],
    "Decisión (α={:.2f})".format(nivel_significancia): [
        "RECHAZA H₀" if p_val_graf < nivel_significancia else "No rechaza H₀",
        "RECHAZA H₀" if p_t_g < nivel_significancia else "No rechaza H₀",
        "Ver gráfica",
        "Ver gráfica"
    ]
}

df_resumen = pd.DataFrame(resumen_data)
st.dataframe(df_resumen, use_container_width=True, hide_index=True)

st.markdown("""
<div class='tarjeta tarjeta-naranja' style='margin-top:16px;'>
    <strong>📌 Nota Metodológica</strong><br>
    Esta aplicación implementa pruebas de hipótesis para diferencia de medias, varianzas y razón de varianzas 
    siguiendo la metodología de <em>Métodos Estadísticos I — INGI02</em>. 
    Ajusta los parámetros en el panel lateral (α, tipo de prueba, σ) para explorar distintos escenarios.
</div>
""", unsafe_allow_html=True)
<div style="text-align:center; color:#999; font-size:0.85rem; margin-top:32px; padding-bottom:20px;">
  Tutor de Estadística Inferencial  FACYT UC  ING102  Proceso de 8 Pasos
</div>
""", unsafe_allow_html=True)
