import streamlit as st
import numpy as np
from scipy import stats
import math

# ─── Configuración de página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Tutor de Estadística Inferencial",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Estilos CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap');

  html, body, [class*="css"] {
    font-family: 'Crimson Pro', Georgia, serif;
  }
  h1, h2, h3 {
    font-family: 'Libre Baskerville', Georgia, serif !important;
    letter-spacing: -0.02em;
  }
  .stApp {
    background: #faf8f3;
  }
  .paso-box {
    background: #fffef7;
    border-left: 4px solid #2c3e50;
    border-radius: 0 6px 6px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 1.05rem;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.06);
  }
  .paso-titulo {
    font-family: 'Libre Baskerville', serif;
    font-weight: 700;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #555;
    margin-bottom: 6px;
  }
  .formula-box {
    background: #f0ede4;
    border: 1px solid #ccc9b8;
    border-radius: 4px;
    padding: 10px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.97rem;
    margin: 6px 0;
  }
  .resultado-destacado {
    background: #2c3e50;
    color: #f9f6ec;
    border-radius: 4px;
    padding: 10px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    margin: 8px 0;
    display: inline-block;
  }
  .conclusion-box {
    background: #eaf4ea;
    border-left: 4px solid #27ae60;
    border-radius: 0 6px 6px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 1.05rem;
  }
  .rechazo-box {
    background: #fdecea;
    border-left: 4px solid #c0392b;
    border-radius: 0 6px 6px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 1.05rem;
  }
  .header-exam {
    background: #2c3e50;
    color: #f9f6ec;
    padding: 24px 32px;
    border-radius: 8px;
    margin-bottom: 28px;
  }
  .table-muestra th {
    background: #2c3e50;
    color: white;
  }
  code {
    font-family: 'JetBrains Mono', monospace !important;
    background: #f0ede4 !important;
    padding: 2px 6px;
    border-radius: 3px;
  }
  .stExpander > div:first-child {
    background: #f5f2e8;
    border-radius: 6px;
  }
</style>
""", unsafe_allow_html=True)

# ─── Funciones auxiliares ─────────────────────────────────────────────────────

def parsear_datos(texto):
    """Parsea datos separados por coma, punto y coma o espacios."""
    import re
    nums = re.split(r'[,;\s]+', texto.strip())
    return [float(x) for x in nums if x]

def calcular_estadisticos(datos):
    arr = np.array(datos)
    media = np.mean(arr)
    var = np.var(arr, ddof=1)
    desv = np.std(arr, ddof=1)
    n = len(arr)
    return media, var, desv, n

def z_critico(alpha, cola='derecha'):
    if cola == 'derecha':
        return stats.norm.ppf(1 - alpha)
    elif cola == 'izquierda':
        return stats.norm.ppf(alpha)
    else:
        return stats.norm.ppf(1 - alpha/2)

def t_critico(alpha, gl, cola='derecha'):
    if cola == 'derecha':
        return stats.t.ppf(1 - alpha, gl)
    elif cola == 'izquierda':
        return stats.t.ppf(alpha, gl)
    else:
        return stats.t.ppf(1 - alpha/2, gl)

def grafico_distribucion(titulo, valor_estadistico, valor_critico, tipo='Z',
                          cola='derecha', x_label=None):
    """Genera SVG de distribución con zona de rechazo."""
    # Rango de x
    lim = max(4.0, abs(valor_estadistico) + 0.8, abs(valor_critico) + 0.8)
    xs = np.linspace(-lim, lim, 400)
    ys = stats.norm.pdf(xs)
    
    W, H = 600, 220
    x_min, x_max = -lim, lim
    y_max = max(ys) * 1.15
    
    def px(v): return (v - x_min) / (x_max - x_min) * W
    def py(v): return H - 20 - v / y_max * (H - 50)
    
    # Polígono curva
    puntos_curva = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    puntos_cierre = f"{px(xs[-1]):.1f},{py(0):.1f} {px(xs[0]):.1f},{py(0):.1f}"
    
    # Zona de rechazo
    if cola == 'derecha':
        xs_rej = xs[xs >= valor_critico]
        ys_rej = stats.norm.pdf(xs_rej)
    elif cola == 'izquierda':
        xs_rej = xs[xs <= valor_critico]
        ys_rej = stats.norm.pdf(xs_rej)
    else:
        xs_rej1 = xs[xs <= -abs(valor_critico)]
        ys_rej1 = stats.norm.pdf(xs_rej1)
        xs_rej2 = xs[xs >= abs(valor_critico)]
        ys_rej2 = stats.norm.pdf(xs_rej2)

    def poligono_rechazo(xr, yr):
        if len(xr) == 0:
            return ""
        puntos = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xr, yr))
        puntos += f" {px(xr[-1]):.1f},{py(0):.1f} {px(xr[0]):.1f},{py(0):.1f}"
        return f'<polygon points="{puntos}" fill="rgba(192,57,43,0.35)" stroke="none"/>'
    
    zona_rechazo_svg = ""
    if cola in ('derecha', 'izquierda'):
        zona_rechazo_svg = poligono_rechazo(xs_rej, ys_rej)
    else:
        zona_rechazo_svg = poligono_rechazo(xs_rej1, ys_rej1) + poligono_rechazo(xs_rej2, ys_rej2)
    
    # Línea estadístico de prueba
    xe = px(valor_estadistico)
    ye_top = py(stats.norm.pdf(valor_estadistico))
    
    # Línea valor crítico
    xvc = px(valor_critico)
    
    # Eje X etiquetas
    ticks = [round(v, 2) for v in [-lim, -lim/2, 0, lim/2, lim]]
    ticks_svg = ""
    for t in ticks:
        ticks_svg += f'<text x="{px(t):.1f}" y="{H-5}" text-anchor="middle" font-size="11" fill="#555">{t}</text>'
        ticks_svg += f'<line x1="{px(t):.1f}" y1="{H-22}" x2="{px(t):.1f}" y2="{H-18}" stroke="#999" stroke-width="1"/>'
    
    svg = f"""
<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" style="font-family: Georgia, serif;">
  <!-- Fondo -->
  <rect width="{W}" height="{H}" fill="#faf8f3" rx="6"/>
  <!-- Título -->
  <text x="{W//2}" y="18" text-anchor="middle" font-size="13" font-weight="bold" fill="#2c3e50">{titulo}</text>
  <!-- Zona aceptación -->
  <polygon points="{puntos_curva} {puntos_cierre}" fill="rgba(44,62,80,0.10)" stroke="none"/>
  <!-- Curva normal -->
  <polyline points="{puntos_curva}" fill="none" stroke="#2c3e50" stroke-width="2"/>
  <!-- Zona rechazo -->
  {zona_rechazo_svg}
  <!-- Eje X -->
  <line x1="0" y1="{py(0):.1f}" x2="{W}" y2="{py(0):.1f}" stroke="#888" stroke-width="1"/>
  {ticks_svg}
  <!-- Valor crítico -->
  <line x1="{xvc:.1f}" y1="25" x2="{xvc:.1f}" y2="{py(0):.1f}" stroke="#c0392b" stroke-width="2" stroke-dasharray="5,3"/>
  <text x="{xvc:.1f}" y="23" text-anchor="middle" font-size="11" fill="#c0392b">Vc*={valor_critico:.4f}</text>
  <!-- Estadístico de prueba -->
  <line x1="{xe:.1f}" y1="{ye_top:.1f}" x2="{xe:.1f}" y2="{py(0):.1f}" stroke="#1a6b3c" stroke-width="2"/>
  <circle cx="{xe:.1f}" cy="{ye_top:.1f}" r="4" fill="#1a6b3c"/>
  <text x="{xe:.1f}" y="{ye_top-8:.1f}" text-anchor="middle" font-size="11" fill="#1a6b3c">Z₀={valor_estadistico:.4f}</text>
  <!-- Leyenda -->
  <rect x="5" y="{H-48}" width="12" height="10" fill="rgba(192,57,43,0.35)"/>
  <text x="20" y="{H-40}" font-size="10" fill="#c0392b">Región de Rechazo</text>
  <line x1="5" y1="{H-28}" x2="17" y2="{H-28}" stroke="#1a6b3c" stroke-width="2"/>
  <text x="20" y="{H-24}" font-size="10" fill="#1a6b3c">Estadístico Z₀</text>
</svg>"""
    return svg


def grafico_chi2(titulo, valor_estadistico, valor_critico, gl):
    """SVG para distribución chi-cuadrado."""
    xs = np.linspace(0.01, max(50, valor_critico * 1.5, valor_estadistico * 1.3), 400)
    ys = stats.chi2.pdf(xs, gl)
    
    W, H = 600, 220
    x_max = max(50, valor_critico * 1.5, valor_estadistico * 1.3)
    y_max = max(ys) * 1.2
    
    def px(v): return v / x_max * (W - 20) + 10
    def py(v): return H - 20 - v / y_max * (H - 50)
    
    puntos_curva = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    puntos_cierre = f"{px(xs[-1]):.1f},{py(0):.1f} {px(xs[0]):.1f},{py(0):.1f}"
    
    xs_rej = xs[xs >= valor_critico]
    ys_rej = stats.chi2.pdf(xs_rej, gl)
    
    zona_rej = ""
    if len(xs_rej) > 0:
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs_rej, ys_rej))
        pts += f" {px(xs_rej[-1]):.1f},{py(0):.1f} {px(xs_rej[0]):.1f},{py(0):.1f}"
        zona_rej = f'<polygon points="{pts}" fill="rgba(192,57,43,0.35)" stroke="none"/>'
    
    xvc = px(valor_critico)
    xe = px(valor_estadistico)
    
    svg = f"""
<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{W}" height="{H}" fill="#faf8f3" rx="6"/>
  <text x="{W//2}" y="18" text-anchor="middle" font-size="13" font-weight="bold" fill="#2c3e50">{titulo}</text>
  <polygon points="{puntos_curva} {puntos_cierre}" fill="rgba(44,62,80,0.10)" stroke="none"/>
  <polyline points="{puntos_curva}" fill="none" stroke="#2c3e50" stroke-width="2"/>
  {zona_rej}
  <line x1="10" y1="{py(0):.1f}" x2="{W-10}" y2="{py(0):.1f}" stroke="#888" stroke-width="1"/>
  <line x1="{xvc:.1f}" y1="25" x2="{xvc:.1f}" y2="{py(0):.1f}" stroke="#c0392b" stroke-width="2" stroke-dasharray="5,3"/>
  <text x="{xvc:.1f}" y="23" text-anchor="middle" font-size="11" fill="#c0392b">χ²c={valor_critico:.3f}</text>
  <line x1="{xe:.1f}" y1="35" x2="{xe:.1f}" y2="{py(0):.1f}" stroke="#1a6b3c" stroke-width="2"/>
  <text x="{xe:.1f}" y="33" text-anchor="middle" font-size="11" fill="#1a6b3c">χ²₀={valor_estadistico:.4f}</text>
  <text x="{W//2}" y="{H-5}" text-anchor="middle" font-size="10" fill="#888">χ²(gl={gl})</text>
</svg>"""
    return svg


def grafico_F(titulo, valor_estadistico, valor_critico, gl1, gl2):
    """SVG para distribución F de Fisher."""
    lim_x = max(valor_critico * 1.6, valor_estadistico * 1.3, 6)
    xs = np.linspace(0.01, lim_x, 400)
    ys = stats.f.pdf(xs, gl1, gl2)
    
    W, H = 600, 220
    y_max = max(ys) * 1.2
    
    def px(v): return v / lim_x * (W - 20) + 10
    def py(v): return H - 20 - v / y_max * (H - 50)
    
    puntos_curva = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    puntos_cierre = f"{px(xs[-1]):.1f},{py(0):.1f} {px(xs[0]):.1f},{py(0):.1f}"
    
    xs_rej = xs[xs >= valor_critico]
    ys_rej = stats.f.pdf(xs_rej, gl1, gl2)
    zona_rej = ""
    if len(xs_rej) > 0:
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs_rej, ys_rej))
        pts += f" {px(xs_rej[-1]):.1f},{py(0):.1f} {px(xs_rej[0]):.1f},{py(0):.1f}"
        zona_rej = f'<polygon points="{pts}" fill="rgba(192,57,43,0.35)" stroke="none"/>'
    
    xvc = px(valor_critico)
    xe = px(valor_estadistico)
    
    svg = f"""
<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{W}" height="{H}" fill="#faf8f3" rx="6"/>
  <text x="{W//2}" y="18" text-anchor="middle" font-size="13" font-weight="bold" fill="#2c3e50">{titulo}</text>
  <polygon points="{puntos_curva} {puntos_cierre}" fill="rgba(44,62,80,0.10)" stroke="none"/>
  <polyline points="{puntos_curva}" fill="none" stroke="#2c3e50" stroke-width="2"/>
  {zona_rej}
  <line x1="10" y1="{py(0):.1f}" x2="{W-10}" y2="{py(0):.1f}" stroke="#888" stroke-width="1"/>
  <line x1="{xvc:.1f}" y1="25" x2="{xvc:.1f}" y2="{py(0):.1f}" stroke="#c0392b" stroke-width="2" stroke-dasharray="5,3"/>
  <text x="{xvc:.1f}" y="23" text-anchor="middle" font-size="11" fill="#c0392b">Fc={valor_critico:.4f}</text>
  <line x1="{xe:.1f}" y1="35" x2="{xe:.1f}" y2="{py(0):.1f}" stroke="#1a6b3c" stroke-width="2"/>
  <text x="{xe:.1f}" y="33" text-anchor="middle" font-size="11" fill="#1a6b3c">F₀={valor_estadistico:.4f}</text>
  <text x="{W//2}" y="{H-5}" text-anchor="middle" font-size="10" fill="#888">F(gl₁={gl1}, gl₂={gl2})</text>
</svg>"""
    return svg


def paso(num, titulo, contenido_html):
    st.markdown(f"""
    <div class="paso-box">
      <div class="paso-titulo">Paso {num} — {titulo}</div>
      {contenido_html}
    </div>
    """, unsafe_allow_html=True)

def formula(tex):
    return f'<div class="formula-box">{tex}</div>'

def resultado(tex):
    return f'<span class="resultado-destacado">{tex}</span>'

# ─── ENCABEZADO ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-exam">
  <h1 style="color:#f9f6ec; margin:0; font-size:1.8rem;">📊 Tutor de Estadística Inferencial</h1>
  <p style="color:#bdc3c7; margin:6px 0 0; font-size:1rem;">
    Prueba de Hipótesis — Proceso de 8 Pasos · FACYT UC · ING102
  </p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR — Datos de Muestra ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Datos de las Muestras")
    
    num_pob = st.number_input("Número de arquitecturas/poblaciones", min_value=1, max_value=4, value=2, step=1)
    
    sigma_conocida = st.checkbox("¿Desviación estándar poblacional (σ) conocida?", value=True)
    
    muestras = []
    for i in range(int(num_pob)):
        st.markdown(f"---\n**Arquitectura {i+1}**")
        nombre = st.text_input(f"Nombre arquitectura {i+1}", value=["SwiftChimb", "SwiftPay", "Arquitectura 3", "Arquitectura 4"][i], key=f"nombre_{i}")
        datos_raw = st.text_area(
            f"Datos muestra {i+1} (separados por coma/espacio)",
            value=["3.30, 3.42, 3.36, 3.27, 3.45, 3.33, 3.39, 3.30, 3.36, 3.24, 3.40, 3.31, 3.38, 3.35, 3.29",
                   "2.05, 2.14, 2.02, 2.08, 2.17, 2.11, 2.05, 2.14, 2.02, 2.08, 2.17, 2.11, 2.09, 1.98, 2.22, 1.95, 2.25, 2.08",
                   "", ""][i],
            height=90, key=f"datos_{i}"
        )
        sigma_val = None
        if sigma_conocida:
            sigma_val = st.number_input(f"σ_{i+1} (desv. poblacional)", min_value=0.0001, value=0.18, format="%.4f", key=f"sigma_{i}")
        
        try:
            datos = parsear_datos(datos_raw) if datos_raw.strip() else []
        except:
            datos = []
            st.warning(f"Error al parsear datos de arquitectura {i+1}")
        
        if datos:
            media, var, desv, n = calcular_estadisticos(datos)
        else:
            media = var = desv = 0.0
            n = st.number_input(f"n_{i+1} (si no hay datos)", min_value=1, value=15, key=f"n_man_{i}")
        
        muestras.append({
            "nombre": nombre,
            "datos": datos,
            "media": media,
            "var": var,
            "desv": desv,
            "n": int(n) if datos else n,
            "sigma": sigma_val,
        })

# ─── PANEL: Tabla de Datos Muestrales ────────────────────────────────────────
st.markdown("## 1. Datos Muestrales")

cols = st.columns(len(muestras))
for i, m in enumerate(muestras):
    with cols[i]:
        st.markdown(f"**Arquitectura {i+1}: {m['nombre']}**")
        simbolos = {
            "X̄": f"{m['media']:.4f} s",
            "s²": f"{m['var']:.6f} s²",
            "s":  f"{m['desv']:.6f} s",
            "n":  f"{m['n']}",
        }
        if m['sigma']:
            simbolos["σ"] = f"{m['sigma']:.4f} s"
        for k, v in simbolos.items():
            st.markdown(f"<div class='formula-box'><b>{k}_{i+1}</b> = {v}</div>", unsafe_allow_html=True)

# Referencias rápidas
X1, X2 = muestras[0]['media'], muestras[1]['media'] if len(muestras) > 1 else 0
s1, s2 = muestras[0]['desv'], muestras[1]['desv'] if len(muestras) > 1 else 0
s2_1, s2_2 = muestras[0]['var'], muestras[1]['var'] if len(muestras) > 1 else 0
n1, n2 = muestras[0]['n'], muestras[1]['n'] if len(muestras) > 1 else 0
sig1 = muestras[0]['sigma'] if muestras[0]['sigma'] else s1
sig2 = muestras[1]['sigma'] if len(muestras) > 1 and muestras[1]['sigma'] else s2

st.markdown("---")
st.markdown("## 2. Incisos del Parcial")

# ════════════════════════════════════════════════════════════════════════════
# INCISO A: Prueba Z de diferencia de medias (σ conocida)
# ════════════════════════════════════════════════════════════════════════════
with st.expander("📌 Inciso a) — Autorización del despliegue (Prueba Z, σ conocida)", expanded=True):
    st.markdown("*El equipo de analistas afirma que la mejora supera la diferencia planteada.*")
    
    col1, col2 = st.columns(2)
    with col1:
        alpha_a = st.number_input("Nivel de significancia α", value=0.05, min_value=0.001, max_value=0.5, format="%.3f", key="alpha_a")
        delta_0_a = st.number_input("Diferencia bajo H₀ (μ₁ − μ₂)₀", value=1.2, format="%.4f", key="delta_a")
    
    st.markdown("### Procedimiento — 8 Pasos")
    
    paso(1, "Parámetro de Interés",
         "<b>Parámetro:</b> μ₁ − μ₂ (diferencia de tiempos promedio entre arquitecturas)")
    
    paso(2, "Hipótesis Nula (H₀)",
         formula(f"H₀: μ₁ − μ₂ = {delta_0_a}"))
    
    paso(3, "Hipótesis Alternativa (H₁)",
         formula(f"H₁: μ₁ − μ₂ > {delta_0_a}") + "<br>→ Prueba <b>unilateral derecha</b>")
    
    paso(4, "Nivel de Significancia",
         formula(f"α = {alpha_a}"))
    
    # Cálculos
    error_std_a = math.sqrt(sig1**2/n1 + sig2**2/n2)
    Z0_a = ((X1 - X2) - delta_0_a) / error_std_a
    Zc_a = z_critico(alpha_a, 'derecha')
    Vc_a = delta_0_a + Zc_a * error_std_a
    valor_p_a = 1 - stats.norm.cdf(Z0_a)
    
    paso(5, "Estadístico de Prueba",
         formula(f"Z₀ = [(X̄₁ − X̄₂) − (μ₁−μ₂)₀] / √(σ₁²/n₁ + σ₂²/n₂)") +
         formula(f"Z₀ = [({X1:.4f} − {X2:.4f}) − {delta_0_a}] / √({sig1:.4f}²/{n1} + {sig2:.4f}²/{n2})") +
         formula(f"Z₀ = [{X1-X2:.4f} − {delta_0_a}] / √({sig1**2/n1:.6f} + {sig2**2/n2:.6f})") +
         formula(f"Z₀ = {(X1-X2-delta_0_a):.4f} / {error_std_a:.4f}") +
         resultado(f"Z₀ = {Z0_a:.4f}"))
    
    paso(6, "Región de Rechazo",
         formula(f"Se rechaza H₀ si Z₀ > Z_(1−α) = Z_{1-alpha_a}") +
         formula(f"Z_(1−α) = Z_{1-alpha_a} = {Zc_a:.4f}") +
         formula(f"Vc* = (μ₁−μ₂)₀ + Z_(1−α) · √(σ₁²/n₁ + σ₂²/n₂)") +
         formula(f"Vc* = {delta_0_a} + {Zc_a:.4f} · {error_std_a:.4f}") +
         resultado(f"Vc* = {Vc_a:.4f} s"))
    
    paso(7, "Cálculo del Valor-p",
         formula(f"Valor-p = P(Z > Z₀ | H₀)") +
         formula(f"Valor-p = P(Z > {Z0_a:.4f})") +
         resultado(f"Valor-p = {valor_p_a:.4f}"))
    
    # Conclusión
    rechaza = Z0_a > Zc_a
    if rechaza:
        st.markdown(f"""
        <div class="conclusion-box">
          <div class="paso-titulo">Paso 8 — Conclusión</div>
          <b>Decisión:</b> Z₀ = {Z0_a:.4f} > Vc* = {Zc_a:.4f} → <b>Se rechaza H₀</b><br>
          Valor-p = {valor_p_a:.4f} < α = {alpha_a} → Confirma el rechazo<br><br>
          <b>Conclusión:</b> Con un nivel de significancia α = {alpha_a}, existe suficiente evidencia
          estadística para afirmar que la diferencia de tiempos (μ₁ − μ₂) supera {delta_0_a} segundos.
          <b>Se autoriza el despliegue nacional de {muestras[1]['nombre']}.</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="rechazo-box">
          <div class="paso-titulo">Paso 8 — Conclusión</div>
          <b>Decisión:</b> Z₀ = {Z0_a:.4f} ≤ Vc* = {Zc_a:.4f} → <b>No se rechaza H₀</b><br>
          Valor-p = {valor_p_a:.4f} ≥ α = {alpha_a}<br><br>
          <b>Conclusión:</b> No existe suficiente evidencia estadística para afirmar que la diferencia
          supera {delta_0_a} segundos con α = {alpha_a}.
          <b>No se autoriza el despliegue nacional.</b>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráfico
    st.markdown("**Gráfico — Distribución Normal Estándar (Inciso a)**")
    svg_a = grafico_distribucion(
        f"Inciso a) — Prueba Z (α={alpha_a}, H₁: μ₁−μ₂ > {delta_0_a})",
        Z0_a, Zc_a, tipo='Z', cola='derecha'
    )
    st.markdown(svg_a, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# INCISO B: Prueba t, varianzas desconocidas pero iguales
# ════════════════════════════════════════════════════════════════════════════
with st.expander("📌 Inciso b) — Varianzas desconocidas pero iguales (Prueba t)"):
    st.markdown("*Si por condiciones de carga se asumen varianzas desconocidas pero iguales, ¿cambia la decisión?*")
    
    col1, col2 = st.columns(2)
    with col1:
        alpha_b = st.number_input("Nivel de significancia α", value=0.05, min_value=0.001, max_value=0.5, format="%.3f", key="alpha_b")
        delta_0_b = st.number_input("Diferencia bajo H₀ (μ₁ − μ₂)₀", value=1.2, format="%.4f", key="delta_b")
    with col2:
        st.info("Se asumen varianzas poblacionales **desconocidas pero iguales**: σ₁² = σ₂²")
    
    st.markdown("### Procedimiento — 8 Pasos")
    
    paso(1, "Parámetro de Interés", "<b>Parámetro:</b> μ₁ − μ₂")
    paso(2, "Hipótesis Nula (H₀)", formula(f"H₀: μ₁ − μ₂ = {delta_0_b}"))
    paso(3, "Hipótesis Alternativa (H₁)", formula(f"H₁: μ₁ − μ₂ > {delta_0_b}") + "<br>→ Prueba <b>unilateral derecha</b>")
    paso(4, "Nivel de Significancia", formula(f"α = {alpha_b}"))
    
    # Varianza agrupada
    gl_b = n1 + n2 - 2
    sp2 = ((n1 - 1) * s2_1 + (n2 - 1) * s2_2) / gl_b
    sp = math.sqrt(sp2)
    error_std_b = sp * math.sqrt(1/n1 + 1/n2)
    t0_b = ((X1 - X2) - delta_0_b) / error_std_b
    tc_b = t_critico(alpha_b, gl_b, 'derecha')
    Vc_b = delta_0_b + tc_b * error_std_b
    valor_p_b = 1 - stats.t.cdf(t0_b, gl_b)
    
    paso(5, "Estadístico de Prueba",
         "<b>Varianza Agrupada (pooled):</b>" +
         formula(f"s²ₚ = [(n₁−1)·s²₁ + (n₂−1)·s²₂] / (n₁+n₂−2)") +
         formula(f"s²ₚ = [({n1}−1)·{s2_1:.6f} + ({n2}−1)·{s2_2:.6f}] / ({n1}+{n2}−2)") +
         formula(f"s²ₚ = [{(n1-1)*s2_1:.6f} + {(n2-1)*s2_2:.6f}] / {gl_b}") +
         resultado(f"s²ₚ = {sp2:.6f}  →  sₚ = {sp:.6f}") +
         "<br><b>Estadístico t₀:</b>" +
         formula(f"t₀ = [(X̄₁ − X̄₂) − (μ₁−μ₂)₀] / [sₚ · √(1/n₁ + 1/n₂)]") +
         formula(f"t₀ = [{X1:.4f} − {X2:.4f} − {delta_0_b}] / [{sp:.6f} · √(1/{n1} + 1/{n2})]") +
         formula(f"t₀ = {X1-X2-delta_0_b:.4f} / {error_std_b:.6f}") +
         resultado(f"t₀ = {t0_b:.4f}") +
         formula(f"Grados de libertad: gl = n₁ + n₂ − 2 = {n1} + {n2} − 2 = {gl_b}"))
    
    paso(6, "Región de Rechazo",
         formula(f"Se rechaza H₀ si t₀ > t_(α, gl) = t_({alpha_b}, {gl_b})") +
         formula(f"t_(α={alpha_b}, gl={gl_b}) = {tc_b:.4f}") +
         formula(f"Vc* = (μ₁−μ₂)₀ + t_(α,gl) · sₚ · √(1/n₁ + 1/n₂)") +
         resultado(f"Vc* = {Vc_b:.4f} s"))
    
    paso(7, "Valor-p",
         formula(f"Valor-p = P(t_({gl_b}) > {t0_b:.4f})") +
         resultado(f"Valor-p = {valor_p_b:.4f}"))
    
    rechaza_b = t0_b > tc_b
    if rechaza_b:
        st.markdown(f"""
        <div class="conclusion-box">
          <div class="paso-titulo">Paso 8 — Conclusión</div>
          <b>Decisión:</b> t₀ = {t0_b:.4f} > t_c = {tc_b:.4f} → <b>Se rechaza H₀</b><br>
          Valor-p = {valor_p_b:.4f} < α = {alpha_b}<br><br>
          <b>La decisión {"cambia" if rechaza_b != (Z0_a > Zc_a) else "no cambia"}</b> con respecto al inciso a).
          Con varianzas desconocidas pero iguales, {'existe' if rechaza_b else 'no existe'} evidencia
          suficiente para autorizar el despliegue.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="rechazo-box">
          <div class="paso-titulo">Paso 8 — Conclusión</div>
          <b>Decisión:</b> t₀ = {t0_b:.4f} ≤ t_c = {tc_b:.4f} → <b>No se rechaza H₀</b><br>
          Valor-p = {valor_p_b:.4f} ≥ α = {alpha_b}<br><br>
          Con varianzas desconocidas e iguales, la decisión {'cambia' if rechaza_b != (Z0_a > Zc_a) else 'no cambia'}.
        </div>
        """, unsafe_allow_html=True)
    
    # Gráfico t
    st.markdown("**Gráfico — Distribución t de Student (Inciso b)**")
    # Usamos normal como aproximación en el gráfico
    svg_b = grafico_distribucion(
        f"Inciso b) — Prueba t (gl={gl_b}, α={alpha_b})",
        t0_b, tc_b, tipo='t', cola='derecha'
    )
    st.markdown(svg_b, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# INCISO C: Potencia de la prueba
# ════════════════════════════════════════════════════════════════════════════
with st.expander("📌 Inciso c) — Probabilidad de apoyar la afirmación (Potencia, 1−β)"):
    st.markdown("*¿Cuál sería la probabilidad de apoyar la afirmación si la verdadera diferencia fuera de 1.35 s?*")
    
    col1, col2 = st.columns(2)
    with col1:
        delta_1_c = st.number_input("Verdadera diferencia μ₁−μ₂ (H₁ verdadera)", value=1.35, format="%.4f", key="delta1_c")
        alpha_c = st.number_input("α (mismo que inciso a)", value=0.05, format="%.3f", key="alpha_c")
    
    # Valor crítico Vc* del inciso a (con mismo alpha)
    error_std_c = math.sqrt(sig1**2/n1 + sig2**2/n2)
    Zc_c = z_critico(alpha_c, 'derecha')
    Vc_c = delta_0_a + Zc_c * error_std_c  # Usa delta_0 del inciso a
    
    st.markdown("### Procedimiento — Cálculo de Potencia")
    
    paso("—", "Referencia: Valor Crítico Vc* del inciso a)",
         formula(f"Vc* = (μ₁−μ₂)₀ + Z_(1−α) · √(σ₁²/n₁ + σ₂²/n₂)") +
         formula(f"Vc* = {delta_0_a} + {Zc_c:.4f} · {error_std_c:.4f}") +
         resultado(f"Vc* = {Vc_c:.4f} s"))
    
    # β = P(no rechazar H0 | H1 es verdadera)
    # = P(X1-X2 < Vc* | μ1-μ2 = delta_1)
    # = P(Z < (Vc* - delta_1) / error_std)
    Z_beta_c = (Vc_c - delta_1_c) / error_std_c
    beta_c = stats.norm.cdf(Z_beta_c)
    potencia_c = 1 - beta_c
    
    paso("—", "Cálculo de β (Error Tipo II)",
         formula(f"β = P(Aceptar H₀ | H₁: μ₁−μ₂ = {delta_1_c})") +
         formula(f"β = P(X̄₁−X̄₂ < Vc* | μ₁−μ₂ = {delta_1_c})") +
         formula(f"β = P( Z < (Vc* − (μ₁−μ₂)₁) / √(σ₁²/n₁ + σ₂²/n₂) )") +
         formula(f"β = P( Z < ({Vc_c:.4f} − {delta_1_c}) / {error_std_c:.4f} )") +
         formula(f"β = P( Z < {Z_beta_c:.4f} )") +
         resultado(f"β = {beta_c:.4f}"))
    
    paso("—", "Potencia de la Prueba (1−β)",
         formula(f"1 − β = 1 − {beta_c:.4f}") +
         resultado(f"1 − β = {potencia_c:.4f}  ≈  {potencia_c*100:.1f}%"))
    
    st.markdown(f"""
    <div class="conclusion-box">
      <b>Conclusión:</b> Si la verdadera diferencia de tiempos es {delta_1_c} s,
      la probabilidad de <b>detectar correctamente</b> que la diferencia supera {delta_0_a} s
      (apoyar la afirmación del equipo de analistas) es del <b>{potencia_c*100:.1f}%</b>.
      El error de tipo II es β = {beta_c:.4f}.
    </div>
    """, unsafe_allow_html=True)
    
    # Gráfico potencia: dos normales
    st.markdown("**Gráfico — Distribuciones bajo H₀ y H₁ (Inciso c)**")
    # Construir SVG de dos curvas
    W, H_svg = 620, 240
    xs_arr = np.linspace(-1, 3.5, 600)
    lim_x_c = 3.5
    def to_px(v): return (v + 1) / 4.5 * (W - 20) + 10
    y_max_c = stats.norm.pdf(0) * 1.15
    def to_py(v): return H_svg - 25 - v / y_max_c * (H_svg - 55)
    
    # curva H0 (centrada en delta_0_a)
    xs_c = np.linspace(-1, 3.5, 600)
    ys_h0 = stats.norm.pdf(xs_c, loc=delta_0_a, scale=error_std_c)
    ys_h1 = stats.norm.pdf(xs_c, loc=delta_1_c, scale=error_std_c)
    y_max_c2 = max(max(ys_h0), max(ys_h1)) * 1.15
    def to_py2(v): return H_svg - 25 - v / y_max_c2 * (H_svg - 55)
    
    pts_h0 = " ".join(f"{to_px(x):.1f},{to_py2(y):.1f}" for x, y in zip(xs_c, ys_h0))
    pts_h1 = " ".join(f"{to_px(x):.1f},{to_py2(y):.1f}" for x, y in zip(xs_c, ys_h1))
    cierre_h0 = f"{to_px(xs_c[-1]):.1f},{to_py2(0):.1f} {to_px(xs_c[0]):.1f},{to_py2(0):.1f}"
    cierre_h1 = cierre_h0
    
    xvc_px = to_px(Vc_c)
    
    # zona beta (H1, izquierda de Vc*)
    xs_beta = xs_c[xs_c <= Vc_c]
    ys_beta = stats.norm.pdf(xs_beta, loc=delta_1_c, scale=error_std_c)
    pts_beta = " ".join(f"{to_px(x):.1f},{to_py2(y):.1f}" for x, y in zip(xs_beta, ys_beta))
    if len(xs_beta) > 0:
        pts_beta += f" {to_px(xs_beta[-1]):.1f},{to_py2(0):.1f} {to_px(xs_beta[0]):.1f},{to_py2(0):.1f}"
    
    # zona alpha (H0, derecha de Vc*)
    xs_alpha_c = xs_c[xs_c >= Vc_c]
    ys_alpha_c = stats.norm.pdf(xs_alpha_c, loc=delta_0_a, scale=error_std_c)
    pts_alpha_c = " ".join(f"{to_px(x):.1f},{to_py2(y):.1f}" for x, y in zip(xs_alpha_c, ys_alpha_c))
    if len(xs_alpha_c) > 0:
        pts_alpha_c += f" {to_px(xs_alpha_c[-1]):.1f},{to_py2(0):.1f} {to_px(xs_alpha_c[0]):.1f},{to_py2(0):.1f}"
    
    axis_y = to_py2(0)
    
    svg_c = f"""
<svg width="{W}" height="{H_svg}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{W}" height="{H_svg}" fill="#faf8f3" rx="6"/>
  <text x="{W//2}" y="18" text-anchor="middle" font-size="13" font-weight="bold" fill="#2c3e50">Inciso c) — Potencia de la Prueba</text>
  <!-- H0 fill -->
  <polygon points="{pts_h0} {cierre_h0}" fill="rgba(44,62,80,0.10)" stroke="none"/>
  <!-- H1 fill -->
  <polygon points="{pts_h1} {cierre_h1}" fill="rgba(39,174,96,0.10)" stroke="none"/>
  <!-- beta zona -->
  {"" if not pts_beta else f'<polygon points="{pts_beta}" fill="rgba(241,196,15,0.40)" stroke="none"/>'}
  <!-- alpha zona -->
  {"" if not pts_alpha_c else f'<polygon points="{pts_alpha_c}" fill="rgba(192,57,43,0.30)" stroke="none"/>'}
  <!-- curvas -->
  <polyline points="{pts_h0}" fill="none" stroke="#2c3e50" stroke-width="2"/>
  <polyline points="{pts_h1}" fill="none" stroke="#1a6b3c" stroke-width="2"/>
  <!-- eje -->
  <line x1="10" y1="{axis_y:.1f}" x2="{W-10}" y2="{axis_y:.1f}" stroke="#888" stroke-width="1"/>
  <!-- Vc* -->
  <line x1="{xvc_px:.1f}" y1="25" x2="{xvc_px:.1f}" y2="{axis_y:.1f}" stroke="#c0392b" stroke-width="2" stroke-dasharray="5,3"/>
  <text x="{xvc_px:.1f}" y="23" text-anchor="middle" font-size="10" fill="#c0392b">Vc*={Vc_c:.3f}</text>
  <!-- labels -->
  <text x="{to_px(delta_0_a):.1f}" y="{axis_y+14:.1f}" text-anchor="middle" font-size="10" fill="#2c3e50">(μ₁−μ₂)₀={delta_0_a}</text>
  <text x="{to_px(delta_1_c):.1f}" y="{axis_y+14:.1f}" text-anchor="middle" font-size="10" fill="#1a6b3c">(μ₁−μ₂)₁={delta_1_c}</text>
  <!-- leyenda -->
  <rect x="5" y="{H_svg-52}" width="12" height="10" fill="rgba(241,196,15,0.40)"/>
  <text x="20" y="{H_svg-44}" font-size="10" fill="#b7950b">β = {beta_c:.4f}</text>
  <rect x="5" y="{H_svg-38}" width="12" height="10" fill="rgba(192,57,43,0.30)"/>
  <text x="20" y="{H_svg-30}" font-size="10" fill="#c0392b">α = {alpha_c}</text>
  <line x1="5" y1="{H_svg-18}" x2="17" y2="{H_svg-18}" stroke="#1a6b3c" stroke-width="2"/>
  <text x="20" y="{H_svg-14}" font-size="10" fill="#1a6b3c">1−β = {potencia_c:.4f}</text>
</svg>"""
    st.markdown(svg_c, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# INCISO D: Tamaño de muestra mínimo
# ════════════════════════════════════════════════════════════════════════════
with st.expander("📌 Inciso d) — Tamaño de Muestra Mínimo"):
    st.markdown("*Calcule el tamaño de muestra necesario para asegurar la potencia deseada.*")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        potencia_d = st.number_input("Potencia deseada (1−β)", value=0.80, min_value=0.5, max_value=0.999, format="%.3f", key="pot_d")
        beta_d = 1 - potencia_d
    with col2:
        delta_1_d = st.number_input("Verdadera diferencia a detectar (μ₁−μ₂)₁", value=1.30, format="%.4f", key="delta1_d")
    with col3:
        alpha_d = st.number_input("α", value=0.05, format="%.3f", key="alpha_d")
        delta_0_d = st.number_input("(μ₁−μ₂)₀ bajo H₀", value=1.2, format="%.4f", key="delta0_d")
    
    Z_alpha_d = z_critico(alpha_d, 'derecha')  # Z_(1-α)
    Z_beta_d = stats.norm.ppf(1 - beta_d)       # Z_(1-β)
    
    # Varianza total = σ₁²/n + σ₂²/n = (σ₁²+σ₂²)/n  (asumiendo n1=n2=n)
    sigma_total_cuad = sig1**2 + sig2**2
    
    st.markdown("### ★ ¿Qué Tamaño de Muestra se Necesita?")
    st.markdown(f"**Parámetros:** α = {alpha_d}, β = {beta_d:.2f}, (μ₁−μ₂)₀ = {delta_0_d}, (μ₁−μ₂)₁ = {delta_1_d}")
    
    paso("—", "Bajo H₀: expresar Vc* en términos de n",
         formula(f"Vc* = (μ₁−μ₂)₀ + Z_(1−α) · √(σ₁²/n + σ₂²/n)") +
         formula(f"Vc* = {delta_0_d} + Z_(1−{alpha_d}) · √({sig1:.4f}²/n + {sig2:.4f}²/n)") +
         formula(f"Vc* = {delta_0_d} + {Z_alpha_d:.4f} · √({sigma_total_cuad:.4f}/n)   ... (I)"))
    
    paso("—", "Bajo H₁: expresar Vc* en términos de n y potencia",
         formula(f"Vc* = (μ₁−μ₂)₁ + Z_β · √(σ₁²/n + σ₂²/n)") +
         formula(f"Vc* = {delta_1_d} + Z_(β={beta_d:.2f}) · √({sigma_total_cuad:.4f}/n)") +
         formula(f"Vc* = {delta_1_d} + ({-Z_beta_d:.4f}) · √({sigma_total_cuad:.4f}/n)   ... (II)") +
         f"<br>Donde Z_β = −Z_(1−β) = −{Z_beta_d:.4f} = {-Z_beta_d:.4f}")
    
    paso("—", "Igualar (I) = (II) y despejar n",
         formula(f"{delta_0_d} + {Z_alpha_d:.4f}·√({sigma_total_cuad:.4f}/n) = {delta_1_d} + ({-Z_beta_d:.4f})·√({sigma_total_cuad:.4f}/n)") +
         formula(f"({Z_alpha_d:.4f} − ({-Z_beta_d:.4f})) · √({sigma_total_cuad:.4f}/n) = {delta_1_d} − {delta_0_d}") +
         formula(f"({Z_alpha_d + Z_beta_d:.4f}) · √({sigma_total_cuad:.4f}/n) = {delta_1_d - delta_0_d:.4f}") +
         formula(f"√({sigma_total_cuad:.4f}/n) = {(delta_1_d - delta_0_d)/(Z_alpha_d + Z_beta_d):.4f}") +
         formula(f"{sigma_total_cuad:.4f}/n = {((delta_1_d - delta_0_d)/(Z_alpha_d + Z_beta_d))**2:.4f}") +
         formula(f"n = {sigma_total_cuad:.4f} / {((delta_1_d - delta_0_d)/(Z_alpha_d + Z_beta_d))**2:.4f}"))
    
    n_d_exacto = sigma_total_cuad / ((delta_1_d - delta_0_d) / (Z_alpha_d + Z_beta_d))**2
    n_d = math.ceil(n_d_exacto)
    
    st.markdown(f"""
    <div style="text-align:center; margin:16px 0;">
      <div class="resultado-destacado" style="font-size:1.4rem; padding:14px 28px;">
        n = ⌈{n_d_exacto:.2f}⌉ = <b>{n_d}</b> observaciones por arquitectura
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="conclusion-box">
      <b>Conclusión:</b> Se requiere un tamaño de muestra de <b>n = {n_d}</b> observaciones
      por cada arquitectura para asegurar una potencia de {potencia_d} (probabilidad del {potencia_d*100:.0f}%
      de detectar una diferencia real de {delta_1_d} s) con α = {alpha_d}.
    </div>
    """, unsafe_allow_html=True)
    
    # Gráfico (dos normales con n calculado)
    st.markdown("**Gráfico — Tamaño de Muestra (Inciso d)**")
    error_std_d_n = math.sqrt(sigma_total_cuad / n_d)
    Vc_d = delta_0_d + Z_alpha_d * error_std_d_n
    Z_est_d = (Vc_d - delta_1_d) / error_std_d_n
    beta_confirmada = stats.norm.cdf(Z_est_d)
    
    xs_d = np.linspace(delta_0_d - 4*error_std_d_n, delta_1_d + 4*error_std_d_n, 500)
    ys_d0 = stats.norm.pdf(xs_d, loc=delta_0_d, scale=error_std_d_n)
    ys_d1 = stats.norm.pdf(xs_d, loc=delta_1_d, scale=error_std_d_n)
    
    Wd, Hd = 620, 230
    x_min_d, x_max_d = xs_d[0], xs_d[-1]
    y_max_d = max(max(ys_d0), max(ys_d1)) * 1.2
    def ppx(v): return (v - x_min_d)/(x_max_d - x_min_d) * (Wd - 20) + 10
    def ppy(v): return Hd - 22 - v/y_max_d * (Hd - 52)
    
    pts_d0 = " ".join(f"{ppx(x):.1f},{ppy(y):.1f}" for x, y in zip(xs_d, ys_d0))
    pts_d1 = " ".join(f"{ppx(x):.1f},{ppy(y):.1f}" for x, y in zip(xs_d, ys_d1))
    c_d = f"{ppx(xs_d[-1]):.1f},{ppy(0):.1f} {ppx(xs_d[0]):.1f},{ppy(0):.1f}"
    
    xvc_d = ppx(Vc_d)
    axis_d = ppy(0)
    
    svg_d = f"""
<svg width="{Wd}" height="{Hd}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{Wd}" height="{Hd}" fill="#faf8f3" rx="6"/>
  <text x="{Wd//2}" y="18" text-anchor="middle" font-size="13" font-weight="bold" fill="#2c3e50">Inciso d) — n={n_d} (Potencia={potencia_d})</text>
  <polygon points="{pts_d0} {c_d}" fill="rgba(44,62,80,0.12)" stroke="none"/>
  <polygon points="{pts_d1} {c_d}" fill="rgba(39,174,96,0.12)" stroke="none"/>
  <polyline points="{pts_d0}" fill="none" stroke="#2c3e50" stroke-width="2"/>
  <polyline points="{pts_d1}" fill="none" stroke="#1a6b3c" stroke-width="2"/>
  <line x1="10" y1="{axis_d:.1f}" x2="{Wd-10}" y2="{axis_d:.1f}" stroke="#888" stroke-width="1"/>
  <line x1="{xvc_d:.1f}" y1="25" x2="{xvc_d:.1f}" y2="{axis_d:.1f}" stroke="#c0392b" stroke-width="2" stroke-dasharray="5,3"/>
  <text x="{xvc_d:.1f}" y="23" text-anchor="middle" font-size="10" fill="#c0392b">Vc*={Vc_d:.3f}</text>
  <text x="{ppx(delta_0_d):.1f}" y="{axis_d+13:.1f}" text-anchor="middle" font-size="9" fill="#2c3e50">μ₀={delta_0_d}</text>
  <text x="{ppx(delta_1_d):.1f}" y="{axis_d+13:.1f}" text-anchor="middle" font-size="9" fill="#1a6b3c">μ₁={delta_1_d}</text>
  <text x="{Wd-5}" y="32" text-anchor="end" font-size="10" fill="#555">n={n_d}/arquitectura</text>
</svg>"""
    st.markdown(svg_d, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# INCISO E: Prueba Chi-cuadrado de varianza
# ════════════════════════════════════════════════════════════════════════════
with st.expander("📌 Inciso e) — Reportes de latencia (Prueba Chi² de Varianza)"):
    st.markdown("*Se sospecha que la varianza de SwiftPay (2) es mayor a 0.005 s².*")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        sigma0_e = st.number_input("Varianza bajo H₀ (σ²₀)", value=0.005, format="%.4f", key="sigma0_e")
    with col2:
        alpha_e = st.number_input("α", value=0.05, format="%.3f", key="alpha_e")
    with col3:
        muestra_e = st.selectbox("Muestra a analizar", options=list(range(len(muestras))),
                                  format_func=lambda i: f"Arquitectura {i+1}: {muestras[i]['nombre']}",
                                  index=1 if len(muestras) > 1 else 0, key="muestra_e")
    
    s2_e = muestras[muestra_e]['var']
    n_e = muestras[muestra_e]['n']
    gl_e = n_e - 1
    
    chi2_0 = (gl_e * s2_e) / sigma0_e
    chi2_c = stats.chi2.ppf(1 - alpha_e, gl_e)
    valor_p_e = 1 - stats.chi2.cdf(chi2_0, gl_e)
    
    st.markdown("### Procedimiento — 8 Pasos")
    
    paso(1, "Parámetro", f"<b>Parámetro:</b> σ²₂ (varianza de {muestras[muestra_e]['nombre']})")
    paso(2, "H₀", formula(f"H₀: σ²₂ = {sigma0_e}"))
    paso(3, "H₁", formula(f"H₁: σ²₂ > {sigma0_e}") + "<br>→ Prueba <b>unilateral derecha</b>")
    paso(4, "α", formula(f"α = {alpha_e}"))
    
    paso(5, "Estadístico de Prueba χ²",
         formula(f"χ²₀ = (n−1) · s²₂ / σ²₀") +
         formula(f"χ²₀ = ({n_e}−1) · {s2_e:.6f} / {sigma0_e}") +
         formula(f"χ²₀ = {gl_e} · {s2_e:.6f} / {sigma0_e}") +
         formula(f"χ²₀ = {gl_e * s2_e:.6f} / {sigma0_e}") +
         resultado(f"χ²₀ = {chi2_0:.4f}") +
         formula(f"Grados de libertad: gl = n − 1 = {n_e} − 1 = {gl_e}"))
    
    paso(6, "Región de Rechazo",
         formula(f"Se rechaza H₀ si χ²₀ > χ²_(1−α, gl) = χ²_({1-alpha_e}, {gl_e})") +
         resultado(f"χ²_(α={alpha_e}, gl={gl_e}) = {chi2_c:.4f}"))
    
    paso(7, "Valor-p",
         formula(f"Valor-p = P(χ²_({gl_e}) > {chi2_0:.4f})") +
         resultado(f"Valor-p = {valor_p_e:.4f}"))
    
    rechaza_e = chi2_0 > chi2_c
    if rechaza_e:
        st.markdown(f"""
        <div class="conclusion-box">
          <div class="paso-titulo">Paso 8 — Conclusión</div>
          <b>Decisión:</b> χ²₀ = {chi2_0:.4f} > χ²_c = {chi2_c:.4f} → <b>Se rechaza H₀</b><br>
          Valor-p = {valor_p_e:.4f} < α = {alpha_e}<br><br>
          <b>Conclusión:</b> Existe evidencia estadística suficiente para confirmar que la varianza
          de {muestras[muestra_e]['nombre']} es mayor a {sigma0_e} s². Los reportes de latencia se confirman.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="rechazo-box">
          <div class="paso-titulo">Paso 8 — Conclusión</div>
          <b>Decisión:</b> χ²₀ = {chi2_0:.4f} ≤ χ²_c = {chi2_c:.4f} → <b>No se rechaza H₀</b><br>
          Valor-p = {valor_p_e:.4f} ≥ α = {alpha_e}<br><br>
          No existe evidencia suficiente para confirmar que la varianza sea mayor a {sigma0_e} s².
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("**Gráfico — Distribución χ² (Inciso e)**")
    svg_e = grafico_chi2(f"Inciso e) — χ² (gl={gl_e}, α={alpha_e})", chi2_0, chi2_c, gl_e)
    st.markdown(svg_e, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# INCISO F: Cociente de varianzas (F de Fisher)
# ════════════════════════════════════════════════════════════════════════════
with st.expander("📌 Inciso f) — Criterio de seguridad (Cociente de Varianzas, Prueba F)"):
    st.markdown("*El depto. de seguridad impone que σ²₂ debe ser al menos 10% mayor que σ²₁.*")
    
    col1, col2 = st.columns(2)
    with col1:
        ratio_0_f = st.number_input("Ratio bajo H₀ (σ²₂/σ²₁)₀", value=1.10, format="%.4f", key="ratio_f")
    with col2:
        alpha_f = st.number_input("α", value=0.05, format="%.3f", key="alpha_f")
    
    gl1_f = n2 - 1  # Numerador: arquitectura 2 (SwiftPay)
    gl2_f = n1 - 1  # Denominador: arquitectura 1 (SwiftChimb)
    
    F0_f = (s2_2 / s2_1) / ratio_0_f
    Fc_f = stats.f.ppf(1 - alpha_f, gl1_f, gl2_f)
    valor_p_f = 1 - stats.f.cdf(F0_f, gl1_f, gl2_f)
    
    st.markdown("### Procedimiento — 8 Pasos")
    
    paso(1, "Parámetro", "<b>Parámetro:</b> σ²₂ / σ²₁ (cociente de varianzas)")
    paso(2, "H₀", formula(f"H₀: σ²₂ / σ²₁ = {ratio_0_f}"))
    paso(3, "H₁", formula(f"H₁: σ²₂ / σ²₁ > {ratio_0_f}") + "<br>→ Prueba <b>unilateral derecha</b>")
    paso(4, "α", formula(f"α = {alpha_f}"))
    
    paso(5, "Estadístico de Prueba F (Fisher)",
         formula(f"F₀ = (s²₂/s²₁) / (σ²₂/σ²₁)₀") +
         formula(f"F₀ = ({s2_2:.6f} / {s2_1:.6f}) / {ratio_0_f}") +
         formula(f"F₀ = {s2_2/s2_1:.4f} / {ratio_0_f}") +
         resultado(f"F₀ = {F0_f:.4f}") +
         formula(f"gl₁ = n₂ − 1 = {n2} − 1 = {gl1_f}") +
         formula(f"gl₂ = n₁ − 1 = {n1} − 1 = {gl2_f}"))
    
    paso(6, "Región de Rechazo",
         formula(f"Se rechaza H₀ si F₀ > F_(α, gl₁, gl₂) = F_({alpha_f}, {gl1_f}, {gl2_f})") +
         resultado(f"F_c = {Fc_f:.4f}"))
    
    paso(7, "Valor-p",
         formula(f"Valor-p = P(F_({gl1_f},{gl2_f}) > {F0_f:.4f})") +
         resultado(f"Valor-p = {valor_p_f:.4f}"))
    
    rechaza_f = F0_f > Fc_f
    if rechaza_f:
        st.markdown(f"""
        <div class="conclusion-box">
          <div class="paso-titulo">Paso 8 — Conclusión</div>
          <b>Decisión:</b> F₀ = {F0_f:.4f} > F_c = {Fc_f:.4f} → <b>Se rechaza H₀</b><br>
          Valor-p = {valor_p_f:.4f} < α = {alpha_f}<br><br>
          Existe evidencia para apoyar el cumplimiento del criterio de seguridad:
          la varianza de {muestras[1]['nombre']} es más de un {int((ratio_0_f-1)*100)}% mayor que la de {muestras[0]['nombre']}.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="rechazo-box">
          <div class="paso-titulo">Paso 8 — Conclusión</div>
          <b>Decisión:</b> F₀ = {F0_f:.4f} ≤ F_c = {Fc_f:.4f} → <b>No se rechaza H₀</b><br>
          Valor-p = {valor_p_f:.4f} ≥ α = {alpha_f}<br><br>
          No existe evidencia suficiente para concluir que σ²₂/σ²₁ > {ratio_0_f}.
          No se apoya el cumplimiento del criterio de seguridad.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("**Gráfico — Distribución F de Fisher (Inciso f)**")
    svg_f = grafico_F(f"Inciso f) — F({gl1_f},{gl2_f}), α={alpha_f}", F0_f, Fc_f, gl1_f, gl2_f)
    st.markdown(svg_f, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# INCISO G: Probabilidad de liberar despliegue con Vc dado
# ════════════════════════════════════════════════════════════════════════════
with st.expander("📌 Inciso g) — Probabilidad de liberar el despliegue con Vc dado"):
    st.markdown("*Si se establece un valor crítico Vc = 1.30 s, ¿cuál es la probabilidad de liberar el despliegue?*")
    
    col1, col2 = st.columns(2)
    with col1:
        Vc_g = st.number_input("Valor crítico de liberación Vc", value=1.30, format="%.4f", key="Vc_g")
    with col2:
        delta_0_g = st.number_input("(μ₁−μ₂)₀ del inciso a)", value=1.2, format="%.4f", key="delta0_g")
    
    error_std_g = math.sqrt(sig1**2/n1 + sig2**2/n2)
    
    st.markdown("### Procedimiento")
    
    paso("—", "Referencia: parámetros del inciso a)",
         formula(f"σ₁ = {sig1:.4f}, σ₂ = {sig2:.4f}, n₁ = {n1}, n₂ = {n2}") +
         formula(f"Error estándar: √(σ₁²/n₁ + σ₂²/n₂) = √({sig1:.4f}²/{n1} + {sig2:.4f}²/{n2}) = {error_std_g:.4f}"))
    
    # P(liberar) = P(X1-X2 > Vc | H0) = P(Z > (Vc - delta_0) / error_std)
    Z_g = (Vc_g - delta_0_g) / error_std_g
    prob_liberar_g = 1 - stats.norm.cdf(Z_g)
    
    paso("—", "Cálculo de la probabilidad de liberación",
         formula(f"P(Liberar) = P(X̄₁ − X̄₂ > Vc | μ₁−μ₂ = (μ₁−μ₂)₀)") +
         formula(f"= P(Z > (Vc − (μ₁−μ₂)₀) / √(σ₁²/n₁ + σ₂²/n₂))") +
         formula(f"= P(Z > ({Vc_g} − {delta_0_g}) / {error_std_g:.4f})") +
         formula(f"= P(Z > {(Vc_g - delta_0_g):.4f} / {error_std_g:.4f})") +
         formula(f"= P(Z > {Z_g:.4f})") +
         resultado(f"P(Liberar) = {prob_liberar_g:.4f}"))
    
    st.markdown(f"""
    <div class="conclusion-box">
      <b>Conclusión:</b> Con un valor crítico de Vc = {Vc_g} s y basándose en los parámetros
      del inciso a), la probabilidad de liberar el despliegue a nivel nacional es de
      <b>{prob_liberar_g:.4f}</b> ({prob_liberar_g*100:.2f}%).
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Gráfico — Probabilidad de Liberación (Inciso g)**")
    svg_g = grafico_distribucion(
        f"Inciso g) — P(Liberar con Vc={Vc_g})",
        Z_g, Z_g, tipo='Z', cola='derecha'
    )
    st.markdown(svg_g, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: Resumen de Fórmulas
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 3. Resumen de Fórmulas Clave (Chuleta)")

fórmulas = {
    "Media muestral": "X̄ = Σxᵢ / n",
    "Varianza muestral": "s² = Σ(xᵢ − X̄)² / (n−1)",
    "Desviación estándar": "s = √s²",
    "Estadístico Z (σ conocida)": "Z₀ = [(X̄₁−X̄₂) − (μ₁−μ₂)₀] / √(σ₁²/n₁ + σ₂²/n₂)",
    "Estadístico t (σ desconocida, varianzas iguales)": "t₀ = [(X̄₁−X̄₂) − (μ₁−μ₂)₀] / [sₚ·√(1/n₁+1/n₂)],  gl = n₁+n₂−2",
    "Varianza agrupada": "s²ₚ = [(n₁−1)s²₁ + (n₂−1)s²₂] / (n₁+n₂−2)",
    "Estadístico Chi-cuadrado": "χ²₀ = (n−1)·s² / σ²₀,  gl = n−1",
    "Estadístico F de Fisher": "F₀ = (s²₁/s²₂) / (σ²₁/σ²₂)₀,  gl₁=n₁−1, gl₂=n₂−1",
    "Valor crítico Vc* (bajo H₀)": "Vc* = (μ₁−μ₂)₀ + Z_(1−α)·√(σ₁²/n₁+σ₂²/n₂)",
    "Valor crítico Vc* (bajo H₁)": "Vc* = (μ₁−μ₂)₁ + Z_β·√(σ₁²/n₁+σ₂²/n₂)",
    "Error Tipo I": "α = P(Rechazar H₀ | H₀ verdadera)",
    "Error Tipo II": "β = P(Aceptar H₀ | H₁ verdadera)",
    "Potencia": "1 − β = P(Rechazar H₀ | H₁ verdadera)",
    "Tamaño de muestra (n₁=n₂=n)": "n = (σ₁²+σ₂²)·(Z_(1−α)+Z_(1−β))² / (μ₁−μ₂)₁−(μ₁−μ₂)₀)²",
    "Z_(1−α) (α=0.05, cola derecha)": "Z₀.₉₅ = 1.645",
    "Z_(β) (β=0.20)": "Z_β = −Z_(1−β) = −0.841",
}

cols_f = st.columns(2)
items = list(fórmulas.items())
mitad = len(items) // 2
for i, (nombre, formula_str) in enumerate(items):
    col = cols_f[0] if i < mitad else cols_f[1]
    with col:
        st.markdown(f"""
        <div class="paso-box" style="margin:5px 0;">
          <div class="paso-titulo">{nombre}</div>
          <div class="formula-box">{formula_str}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:#999; font-size:0.85rem; margin-top:32px; padding-bottom:20px;">
  Tutor de Estadística Inferencial · FACYT UC · ING102 · Proceso de 8 Pasos
</div>
""", unsafe_allow_html=True)