import numpy as np
import matplotlib.pyplot as plt
import os
# ----------------------------------------------------
# 1. Reconstrucción vía FFT/IFFT  (tu función original)
# ----------------------------------------------------
def reconstruct(signal, N):
    X = np.fft.fft(signal, N)
    x_rec = np.fft.ifft(X)
    return np.real(x_rec)

# ----------------------------------------------------
# 2. Métricas en dominio del tiempo
# ----------------------------------------------------
def time_metrics(x_orig, x_rec):
    """
    Calcula métricas de error entre señal original y reconstruida.
    Retorna un dict con MSE, RMSE, MAXERR, SNR, CORRELACIÓN.
    """
    L = min(len(x_orig), len(x_rec))
    x_rec = x_rec[:L]
    err = x_orig[:L] - x_rec

    mse = np.mean(err**2)
    rmse = np.sqrt(mse)
    maxerr = np.max(np.abs(err))
    power = np.mean(x_orig[:L]**2)
    snr = np.inf if mse == 0 else 10 * np.log10(power / mse)
    corr = np.corrcoef(x_orig[:L], x_rec)[0, 1]

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAXERR": maxerr,
        "SNR_dB": snr,
        "Corr": corr,
    }

# ----------------------------------------------------
# 3. Métricas en dominio de la frecuencia
# ----------------------------------------------------
def freq_metrics(signal, N, fs):
    """
    Calcula métricas del espectro:
    - delta_f
    - frecuencia del pico
    - magnitud del pico
    """
    X = np.fft.fft(signal, N)
    freqs = np.fft.fftfreq(N, 1/fs)

    # Solo parte positiva
    half = N // 2
    mag = np.abs(X[:half])
    f = freqs[:half]

    idx_peak = np.argmax(mag)
    f_peak = f[idx_peak]
    A_peak = mag[idx_peak]

    return {
        "delta_f": fs / N,
        "f_peak": f_peak,
        "A_peak": A_peak,
    }

# ----------------------------------------------------
# 4. Función completa de comparación
# ----------------------------------------------------
def compare_all_metrics(x_orig, fs, N):
    """
    Toma una señal original y calcula todas las métricas relevantes
    para ese N.
    """
    x_rec = reconstruct(x_orig, N)

    tm = time_metrics(x_orig, x_rec)
    fm = freq_metrics(x_orig, N, fs)

    # Unir métricas
    metrics = {
        "N": N,
        **tm,
        **fm
    }
    return metrics
def make_window(N, window_type="hann"):
    """
    Devuelve una ventana de tamaño N.
    window_type: "hann" o "rect"
    """
    if window_type == "hann":
        return np.hanning(N)
    elif window_type == "rect":
        return np.ones(N)
    else:
        raise ValueError("Tipo de ventana no reconocido: usa 'hann' o 'rect'.")


# ===========================================
#     CORRECCIÓN DE AMPLITUD DE LA VENTANA
# ===========================================
def window_gain(window):
    """
    Retorna la ganancia promedio de la ventana para corregir amplitud.
    """
    return np.sum(window) / len(window)


# ===============================
#      ANÁLISIS ESPECTRAL
# ===============================
def spectral_analysis(signal, fs, N, window_type="hann"):
    """
    Devuelve:
      w_norm : frecuencia normalizada [-π, π]
      X_corr : FFT centrada y corregida en amplitud
    """

    # 1. Recortar señal
    xN = signal[:N]

    # 2. Obtener ventana
    w = make_window(N, window_type)

    # 3. Señal enventanada
    xw = xN * w

    # 4. FFT + shift
    X = np.fft.fftshift(np.fft.fft(xw, N))

    # 5. Frecuencia normalizada
    w_norm = np.linspace(-np.pi, np.pi, N)

    # 6. Corrección de amplitud
    gain = window_gain(w)
    X_corr = X / gain / (N/2)

    return w_norm, X_corr


# ===============================
#     GRÁFICOS ESPECTRALES
# ===============================

import re

def plot_spectrum(w_norm, X_corr, title="Spectrum"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(base_dir, "figuras")
    os.makedirs(fig_dir, exist_ok=True)

    # --- FFT centrada ya viene lista ---
    # w_norm ∈ [-π, π]
    # X_corr es el espectro corregido en amplitud

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    # Magnitud
    axes[0].plot(w_norm, np.abs(X_corr))
    axes[0].set_title("Magnitude |X(ω)|")
    axes[0].set_xlabel("Normalized frequency (rad)")
    axes[0].set_ylabel("Magnitude")
    axes[0].grid(True)

    # Parte real
    axes[1].plot(w_norm, np.real(X_corr))
    axes[1].set_title("Real part")
    axes[1].set_xlabel("Normalized frequency (rad)")
    axes[1].set_ylabel("Re{X(ω)}")
    axes[1].grid(True)

    # Parte imaginaria
    axes[2].plot(w_norm, np.imag(X_corr))
    axes[2].set_title("Imaginary part")
    axes[2].set_xlabel("Normalized frequency (rad)")
    axes[2].set_ylabel("Im{X(ω)}")
    axes[2].grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Generar nombre limpio
    fname = "spectrum_" + title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "") + ".png"
    fpath = os.path.join(fig_dir, fname)

    plt.savefig(fpath, dpi=300)
    plt.close()
