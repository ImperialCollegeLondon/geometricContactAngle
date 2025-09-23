import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Iterator, Tuple

_RX_POINT = re.compile(r"^\s*(\d+)\s+(\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+\|\s+(-?\d+(?:\.\d+)?)\s*$")

def iter_contact_angles(path: str, encoding: str = "utf-8") -> Iterator[float]:
    rx = _RX_POINT
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        for ln in f:
            m = rx.match(ln)
            if m:
                yield float(m.group(6))

def read_contact_angles(path: str, as_numpy: bool = True, encoding: str = "utf-8"):
    gen = iter_contact_angles(path, encoding=encoding)
    return np.fromiter(gen, dtype=np.float64) if as_numpy else list(gen)

def gaussian(x: np.ndarray, amp: float, mu: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-(x - mu)**2 / (2.0 * sigma**2))

def _robust_mu_sigma(y: np.ndarray) -> Tuple[float, float]:
    mu0 = float(np.median(y))
    mad = float(np.median(np.abs(y - mu0))) or 1e-6
    sigma0 = 1.4826 * mad
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = float(np.std(y)) or 1.0
    return mu0, sigma0

def fit_gaussian_to_hist(y: np.ndarray, bins: int = 128, hist_range: Tuple[float, float] = (0.0, 180.0)):
    hist, edges = np.histogram(y, bins=bins, range=hist_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    amp0 = float(np.max(hist)) if hist.size else 1.0
    mu0, sigma0 = _robust_mu_sigma(y)
    p0 = [amp0, mu0, max(sigma0, 1e-3)]
    bounds = ([0.0, hist_range[0], 1e-6], [np.inf, hist_range[1], np.inf])
    try:
        params, cov = curve_fit(gaussian, centers, hist, p0=p0, bounds=bounds, maxfev=20000)
    except Exception:
        params, cov = (p0, np.full((3, 3), np.nan))
    return centers, hist, tuple(float(x) for x in params), cov

def plot_contact_angle_histogram_with_fit(path: str,
                                          out_png: str,
                                          bins: int = 128,
                                          hist_range: Tuple[float, float] = (0.0, 180.0),
                                          show_mean_line: bool = True,
                                          show_fit: bool = True):
    y = read_contact_angles(path)
    y = y[(y >= hist_range[0]) & (y <= hist_range[1])]
    if y.size == 0:
        raise ValueError("No contact-angle rows found in file within 0–180°.")
    n = int(y.size)
    mu_s = float(np.mean(y))
    sig_s = float(np.std(y))
    centers, hist, (amp, mu_fit, sigma_fit), cov = fit_gaussian_to_hist(y, bins=bins, hist_range=hist_range)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y, bins=bins, range=hist_range, color="darkorange", edgecolor="black", linewidth=0.6, alpha=0.8,
            label=f"Histogram (n={n}, μ={mu_s:.2f}°, σ={sig_s:.2f}°)")
    if show_fit and np.isfinite(sigma_fit) and sigma_fit > 0:
        xfit = np.linspace(hist_range[0], hist_range[1], 1000)
        yfit = gaussian(xfit, amp, mu_fit, sigma_fit)
        ax.plot(xfit, yfit, linewidth=2.0, label=f"Gaussian fit (μ={mu_fit:.2f}°, σ={sigma_fit:.2f}°)")
    if show_mean_line:
        ax.axvline(mu_s, linestyle="--", linewidth=2, color = "black",label="Sample mean")
    ax.set_xlim(hist_range)
    ax.set_xlabel("Contact Angle (°)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Contact Angle Distribution")
    ax.legend(loc="best", frameon=True, fancybox=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    return (mu_fit, sigma_fit, amp)

file_path = "MW_fw0_CAdata.txt"
out_png = "MW_fw0_CAHistogram.png"

mu, sigma, amp = plot_contact_angle_histogram_with_fit(file_path,out_png)
