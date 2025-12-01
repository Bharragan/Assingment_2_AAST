import pandas as pd
import numpy as np


def ensure_mono(x):
    """Convierte estéreo → mono."""
    if x.ndim == 2:
        return x.mean(axis=1)
    return x

def match_len(a, b):
    L = min(len(a), len(b))
    return a[:L], b[:L]

def snr(ref, est):
    ref, est = match_len(ref, est)
    noise = ref - est
    return 10 * np.log10(np.sum(ref**2) / np.sum(noise**2))

def sisdr(ref, est):
    ref, est = match_len(ref, est)
    alpha = np.dot(est, ref) / np.dot(ref, ref)
    s_target = alpha * ref
    e_noise = est - s_target
    return 10 * np.log10(np.sum(s_target**2) / np.sum(e_noise**2))

def ensure_mono(x):
    """Convierte estéreo → mono."""
    if x.ndim == 2:
        return x.mean(axis=1)
    return x

def match_len(a, b):
    L = min(len(a), len(b))
    return a[:L], b[:L]

def snr(ref, est):
    ref, est = match_len(ref, est)
    noise = ref - est
    return 10 * np.log10(np.sum(ref**2) / np.sum(noise**2))

def sisdr(ref, est):
    ref, est = match_len(ref, est)
    alpha = np.dot(est, ref) / np.dot(ref, ref)
    s_target = alpha * ref
    e_noise = est - s_target
    return 10 * np.log10(np.sum(s_target**2) / np.sum(e_noise**2))