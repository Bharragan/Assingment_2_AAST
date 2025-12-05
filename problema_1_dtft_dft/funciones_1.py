import numpy as np
import matplotlib.pyplot as plt
import os

def reconstruct(signal, N):
    X = np.fft.fft(signal, N)
    x_rec = np.fft.ifft(X)
    return np.real(x_rec)

def time_metrics(x_orig, x_rec):
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

def freq_metrics(signal, N, fs):
    X = np.fft.fft(signal, N)
    freqs = np.fft.fftfreq(N, 1/fs)
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

def compare_all_metrics(x_orig, fs, N):
    x_rec = reconstruct(x_orig, N)

    tm = time_metrics(x_orig, x_rec)
    fm = freq_metrics(x_orig, N, fs)

    metrics = {
        "N": N,
        **tm,
        **fm
    }
    return metrics

def make_window(N, window_type="hann"):
    if window_type == "hann":
        return np.hanning(N)
    elif window_type == "rect":
        return np.ones(N)
    else:
        raise ValueError("Tipo de ventana no reconocido: usa 'hann' o 'rect'.")

def window_gain(window):
    return np.sum(window) / len(window)

def spectral_analysis(signal, fs, N, window_type="hann"):
    xN = signal[:N]
    w = make_window(N, window_type)
    xw = xN * w
    X = np.fft.fftshift(np.fft.fft(xw, N))
    w_norm = np.linspace(-np.pi, np.pi, N)
    gain = window_gain(w)
    X_corr = X / gain / (N/2)
    return w_norm, X_corr

import re

def plot_spectrum(w_norm, X_corr, title="Spectrum"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(base_dir, "figuras")
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].plot(w_norm, np.abs(X_corr))
    axes[0].set_title("Magnitude |X(ω)|")
    axes[0].set_xlabel("Normalized frequency (rad)")
    axes[0].set_ylabel("Magnitude")
    axes[0].grid(True)

    axes[1].plot(w_norm, np.real(X_corr))
    axes[1].set_title("Real part")
    axes[1].set_xlabel("Normalized frequency (rad)")
    axes[1].set_ylabel("Re{X(ω)}")
    axes[1].grid(True)

    axes[2].plot(w_norm, np.imag(X_corr))
    axes[2].set_title("Imaginary part")
    axes[2].set_xlabel("Normalized frequency (rad)")
    axes[2].set_ylabel("Im{X(ω)}")
    axes[2].grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname = "spectrum_" + title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "") + ".png"
    fpath = os.path.join(fig_dir, fname)
    plt.savefig(fpath, dpi=300)
    plt.close()
