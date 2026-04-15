from flask import Flask, render_template, request, jsonify
import math
from scipy import stats
import numpy as np

aplicacion = Flask(__name__)

@aplicacion.route('/')
def inicio():
    return render_template('index.html')

@aplicacion.route('/calcular', methods=['POST'])
def calcular():
    datos = request.get_json()
    operacion = datos.get('operacion')
    
    try:
        if operacion == 'media_aritmetica':
            valores = list(map(float, datos['valores'].replace(',', ' ').split()))
            n = len(valores)
            media = sum(valores) / n
            resultado = {
                'valor': round(media, 6),
                'formula': 'x̄ = Σxᵢ / n',
                'pasos': f'Suma = {sum(valores)}, n = {n}, x̄ = {sum(valores)}/{n} = {round(media, 6)}'
            }

        elif operacion == 'varianza_poblacional':
            valores = list(map(float, datos['valores'].replace(',', ' ').split()))
            media = float(datos['media'])
            n = len(valores)
            suma_cuad = sum((x - media) ** 2 for x in valores)
            varianza = suma_cuad / n
            resultado = {
                'valor': round(varianza, 6),
                'formula': 'σ² = Σ(xᵢ - μ)² / n',
                'pasos': f'Σ(xᵢ - μ)² = {round(suma_cuad, 6)}, n = {n}, σ² = {round(suma_cuad,6)}/{n} = {round(varianza, 6)}'
            }

        elif operacion == 'estadistico_z_diferencia':
            media1 = float(datos['media1'])
            media2 = float(datos['media2'])
            mu1 = float(datos['mu1'])
            mu2 = float(datos['mu2'])
            sigma1_cuad = float(datos['sigma1_cuad'])
            sigma2_cuad = float(datos['sigma2_cuad'])
            n1 = float(datos['n1'])
            n2 = float(datos['n2'])
            numerador = (media1 - media2) - (mu1 - mu2)
            denominador = math.sqrt(sigma1_cuad / n1 + sigma2_cuad / n2)
            z0 = numerador / denominador
            resultado = {
                'valor': round(z0, 6),
                'formula': 'Z₀ = [(x̄₁ - x̄₂) - (μ₁ - μ₂)] / √(σ₁²/n₁ + σ₂²/n₂)',
                'pasos': f'Numerador = ({media1} - {media2}) - ({mu1} - {mu2}) = {round(numerador,6)}\nDenominador = √({sigma1_cuad}/{n1} + {sigma2_cuad}/{n2}) = {round(denominador,6)}\nZ₀ = {round(z0,6)}'
            }

        elif operacion == 'valor_critico_z':
            alfa = float(datos['alfa'])
            cola = datos.get('cola', 'dos')
            if cola == 'derecha':
                z_critico = stats.norm.ppf(1 - alfa)
                descripcion = f'Z_(1-α) = Z_(1-{alfa})'
            elif cola == 'izquierda':
                z_critico = stats.norm.ppf(alfa)
                descripcion = f'Z_α = Z_{alfa}'
            else:
                z_critico = stats.norm.ppf(1 - alfa / 2)
                descripcion = f'Z_(1-α/2) = Z_(1-{alfa}/2)'
            resultado = {
                'valor': round(z_critico, 6),
                'formula': 'Valor Crítico Z₁₋α',
                'pasos': f'α = {alfa}, Cola = {cola}\n{descripcion} = {round(z_critico, 6)}'
            }

        elif operacion == 'estadistico_t_media':
            media_muestral = float(datos['media_muestral'])
            mu = float(datos['mu'])
            s = float(datos['s'])
            n = float(datos['n'])
            t0 = (media_muestral - mu) / (s / math.sqrt(n))
            gl = int(n) - 1
            resultado = {
                'valor': round(t0, 6),
                'formula': 't₀ = (x̄ - μ) / (S/√n)',
                'pasos': f't₀ = ({media_muestral} - {mu}) / ({s}/√{n})\n= {round(media_muestral-mu,6)} / {round(s/math.sqrt(n),6)}\n= {round(t0,6)}\nGrados de libertad: gl = {gl}'
            }

        elif operacion == 'valor_critico_t':
            alfa = float(datos['alfa'])
            gl = int(datos['gl'])
            cola = datos.get('cola', 'dos')
            if cola == 'derecha':
                t_critico = stats.t.ppf(1 - alfa, gl)
            elif cola == 'izquierda':
                t_critico = stats.t.ppf(alfa, gl)
            else:
                t_critico = stats.t.ppf(1 - alfa / 2, gl)
            resultado = {
                'valor': round(t_critico, 6),
                'formula': 'Valor Crítico t_(α, gl)',
                'pasos': f'α = {alfa}, gl = {gl}, Cola = {cola}\nt_critico = {round(t_critico, 6)}'
            }

        elif operacion == 'estadistico_z_media':
            media_muestral = float(datos['media_muestral'])
            mu = float(datos['mu'])
            sigma = float(datos['sigma'])
            n = float(datos['n'])
            z = (media_muestral - mu) / (sigma / math.sqrt(n))
            resultado = {
                'valor': round(z, 6),
                'formula': 'Z = (x̄ - μ) / (σ/√n)',
                'pasos': f'Z = ({media_muestral} - {mu}) / ({sigma}/√{n})\n= {round(media_muestral-mu,6)} / {round(sigma/math.sqrt(n),6)}\n= {round(z,6)}'
            }

        elif operacion == 'valor_critico_z_alfa':
            alfa = float(datos['alfa'])
            z_alfa = stats.norm.ppf(1 - alfa)
            resultado = {
                'valor': round(z_alfa, 6),
                'formula': 'Zα → Valor Crítico de la distribución normal',
                'pasos': f'α = {alfa}\nZ_(1-α) = Z_(1-{alfa}) = {round(z_alfa,6)}'
            }

        elif operacion == 'estadistico_t_diferencia':
            media1 = float(datos['media1'])
            media2 = float(datos['media2'])
            mu1 = float(datos['mu1'])
            mu2 = float(datos['mu2'])
            s1_cuad = float(datos['s1_cuad'])
            s2_cuad = float(datos['s2_cuad'])
            n1 = float(datos['n1'])
            n2 = float(datos['n2'])
            numerador = (media1 - media2) - (mu1 - mu2)
            denominador = math.sqrt(s1_cuad / n1 + s2_cuad / n2)
            t0 = numerador / denominador
            gl = int(((s1_cuad/n1 + s2_cuad/n2)**2) / ((s1_cuad/n1)**2/(n1-1) + (s2_cuad/n2)**2/(n2-1)))
            resultado = {
                'valor': round(t0, 6),
                'formula': 't₀ = [(x̄₁ - x̄₂) - (μ₁ - μ₂)] / √(S₁²/n₁ + S₂²/n₂)',
                'pasos': f'Numerador = ({media1}-{media2}) - ({mu1}-{mu2}) = {round(numerador,6)}\nDenominador = √({s1_cuad}/{n1} + {s2_cuad}/{n2}) = {round(denominador,6)}\nt₀ = {round(t0,6)}\ngl (Welch) ≈ {gl}'
            }

        elif operacion == 'varianza_muestral':
            valores = list(map(float, datos['valores'].replace(',', ' ').split()))
            media = sum(valores) / len(valores)
            n = len(valores)
            suma_cuad = sum((x - media) ** 2 for x in valores)
            s_cuad = suma_cuad / (n - 1)
            resultado = {
                'valor': round(s_cuad, 6),
                'formula': 'S² = Σ(x̄ᵢ - x̄₂)² / (n-1)',
                'pasos': f'x̄ = {round(media,6)}\nΣ(xᵢ - x̄)² = {round(suma_cuad,6)}\nn-1 = {n-1}\nS² = {round(suma_cuad,6)}/{n-1} = {round(s_cuad,6)}'
            }

        elif operacion == 'valor_p':
            estadistico = float(datos['estadistico'])
            tipo = datos.get('tipo', 'z')
            cola = datos.get('cola', 'dos')
            gl = int(datos.get('gl', 30))
            
            if tipo == 'z':
                if cola == 'derecha':
                    p = 1 - stats.norm.cdf(estadistico)
                elif cola == 'izquierda':
                    p = stats.norm.cdf(estadistico)
                else:
                    p = 2 * (1 - stats.norm.cdf(abs(estadistico)))
            else:
                if cola == 'derecha':
                    p = 1 - stats.t.cdf(estadistico, gl)
                elif cola == 'izquierda':
                    p = stats.t.cdf(estadistico, gl)
                else:
                    p = 2 * (1 - stats.t.cdf(abs(estadistico), gl))
            resultado = {
                'valor': round(p, 6),
                'formula': 'Valor p = P(Z > Z₀)',
                'pasos': f'Estadístico = {estadistico}, Tipo = {tipo.upper()}, Cola = {cola}\nValor p = {round(p,6)}'
            }

        elif operacion == 'error_estandar':
            sigma1_cuad = float(datos['sigma1_cuad'])
            sigma2_cuad = float(datos['sigma2_cuad'])
            n1 = float(datos['n1'])
            n2 = float(datos['n2'])
            error = math.sqrt(sigma1_cuad / n1 + sigma2_cuad / n2)
            resultado = {
                'valor': round(error, 6),
                'formula': 'Error Estándar = √(σ₁²/n₁ + σ₂²/n₂)',
                'pasos': f'√({sigma1_cuad}/{n1} + {sigma2_cuad}/{n2})\n= √({round(sigma1_cuad/n1,6)} + {round(sigma2_cuad/n2,6)})\n= {round(error,6)}'
            }

        elif operacion == 'potencia_prueba':
            vc = float(datos['vc'])
            mu1 = float(datos['mu1'])
            mu2 = float(datos['mu2'])
            error_estandar = float(datos['error_estandar'])
            argumento = (vc - (mu1 - mu2)) / error_estandar
            potencia = 1 - stats.norm.cdf(argumento)
            resultado = {
                'valor': round(potencia, 6),
                'formula': '1-β = P(Z > (Vc* - (μ₁-μ₂)) / Error Estándar)',
                'pasos': f'Argumento = ({vc} - ({mu1}-{mu2})) / {error_estandar} = {round(argumento,6)}\n1-β = P(Z > {round(argumento,6)}) = {round(potencia,6)}'
            }

        elif operacion == 'tamano_muestra':
            z1_alfa = float(datos['z1_alfa'])
            z_beta = float(datos['z_beta'])
            sigma1_cuad = float(datos['sigma1_cuad'])
            sigma2_cuad = float(datos['sigma2_cuad'])
            diferencia0 = float(datos['diferencia0'])
            diferencia_a = float(datos['diferencia_a'])
            
            factor_z = z1_alfa + z_beta
            factor_sigma = math.sqrt(sigma1_cuad + sigma2_cuad)
            diferencia = abs(diferencia0 - diferencia_a)
            n = math.ceil(((factor_z * factor_sigma) / diferencia) ** 2)
            resultado = {
                'valor': n,
                'formula': 'n: (Z₁₋α + Zβ) · √(σ₁²/n₁ + σ₂²/n₂) = |(μ₁-μ₂)₀ - (μ₁-μ₂)ₐ|',
                'pasos': f'Z₁₋α = {z1_alfa}, Zβ = {z_beta}\nFactor Z = {z1_alfa}+{z_beta} = {round(factor_z,6)}\nFactor σ = √({sigma1_cuad}+{sigma2_cuad}) = {round(factor_sigma,6)}\nDiferencia = |{diferencia0}-{diferencia_a}| = {diferencia}\nn = ⌈({round(factor_z*factor_sigma,6)}/{diferencia})²⌉ = {n}'
            }

        elif operacion == 'chi_cuadrado':
            s_cuad = float(datos['s_cuad'])
            n = float(datos['n'])
            sigma_cuad = float(datos['sigma_cuad'])
            chi2 = (s_cuad * (n - 1)) / sigma_cuad
            gl = int(n) - 1
            resultado = {
                'valor': round(chi2, 6),
                'formula': 'χ² = S²(n-1) / σ²',
                'pasos': f'χ² = {s_cuad} × ({n}-1) / {sigma_cuad}\n= {s_cuad} × {n-1} / {sigma_cuad}\n= {round(chi2,6)}\nGrados de libertad = {gl}'
            }

        elif operacion == 'estadistico_f':
            s1_cuad = float(datos['s1_cuad'])
            s2_cuad = float(datos['s2_cuad'])
            sigma1_cuad = float(datos.get('sigma1_cuad', 1))
            sigma2_cuad = float(datos.get('sigma2_cuad', 1))
            f = (s1_cuad / sigma1_cuad) / (s2_cuad / sigma2_cuad)
            resultado = {
                'valor': round(f, 6),
                'formula': 'F = (S₁²/σ₁²) / (S₂²/σ₂²) = S₂²/S₁²',
                'pasos': f'F = ({s1_cuad}/{sigma1_cuad}) / ({s2_cuad}/{sigma2_cuad})\n= {round(s1_cuad/sigma1_cuad,6)} / {round(s2_cuad/sigma2_cuad,6)}\n= {round(f,6)}'
            }

        elif operacion == 'valor_critico_f':
            alfa = float(datos['alfa'])
            gl1 = int(datos['gl1'])
            gl2 = int(datos['gl2'])
            f_critico = stats.f.ppf(1 - alfa, gl1, gl2)
            resultado = {
                'valor': round(f_critico, 6),
                'formula': 'Valor F = F_(1-α, gl₁, gl₂)',
                'pasos': f'α = {alfa}, gl₁ = {gl1}, gl₂ = {gl2}\nF_(1-{alfa},{gl1},{gl2}) = {round(f_critico,6)}'
            }

        elif operacion == 'probabilidad_diferencia':
            mu1_mu2_0 = float(datos['mu1_mu2_0'])
            mu1_mu2_1 = float(datos['mu1_mu2_1'])
            sigma1_cuad = float(datos['sigma1_cuad'])
            sigma2_cuad = float(datos['sigma2_cuad'])
            n1 = float(datos['n1'])
            n2 = float(datos['n2'])
            error = math.sqrt(sigma1_cuad / n1 + sigma2_cuad / n2)
            z = (mu1_mu2_0 - mu1_mu2_1) / error
            prob = 1 - stats.norm.cdf(z)
            resultado = {
                'valor': round(prob, 6),
                'formula': 'P(Z > [(μ₁-μ₂)₀ - (μ₁-μ₂)₁] / √(σ₁²/n₁ + σ₂²/n₂))',
                'pasos': f'Error estándar = √({sigma1_cuad}/{n1} + {sigma2_cuad}/{n2}) = {round(error,6)}\nZ = ({mu1_mu2_0} - {mu1_mu2_1}) / {round(error,6)} = {round(z,6)}\nP(Z > {round(z,6)}) = {round(prob,6)}'
            }

        else:
            return jsonify({'error': 'Operación no reconocida'}), 400

        return jsonify({'exito': True, 'resultado': resultado})

    except Exception as e:
        return jsonify({'exito': False, 'error': str(e)}), 400
import streamlit as st
import pandas as pd
import scipy.stats as stats # Para estadística inferencial

st.title("Análisis de Estadística Inferencial")

# 1. Entrada de datos (puedes usar un archivo o inputs)
datos = st.file_uploader("Sube tu dataset (CSV)", type="csv")

if datos:
    df = pd.read_csv(datos)
    st.write("Vista previa de los datos:", df.head())

    # 2. Ejemplo de Prueba de Hipótesis (Inferencial)
    st.subheader("Prueba t de Student")
    col = st.selectbox("Selecciona la variable a analizar", df.columns)
    
    media_hipotetica = st.number_input("Media a contrastar (H0)", value=0.0)
    
    t_stat, p_value = stats.ttest_1samp(df[col].dropna(), media_hipotetica)
    
    # 3. Mostrar resultados profesionales
    st.metric("Estadístico t", f"{t_stat:.4f}")
    st.metric("P-valor", f"{p_value:.4f}")

    if p_value < 0.05:
        st.error("Rechazamos la hipótesis nula (H0): Hay evidencia significativa.")
    else:
        st.success("No se rechaza la hipótesis nula (H0): No hay evidencia suficiente.")
