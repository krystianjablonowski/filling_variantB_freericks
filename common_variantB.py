from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


integrate_trapz = getattr(np, "trapezoid", np.trapz)


@dataclass
class DMFTResult:
    omega: np.ndarray
    rho_arith: np.ndarray
    rho_typ: np.ndarray
    rho_medium: np.ndarray
    green_medium: np.ndarray
    sigma: np.ndarray
    hybrid: np.ndarray
    mu: float
    ne_actual: float
    iterations: int
    converged: bool
    diff: float
    runtime_sec: float


def fmt(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return "True" if x else "False"
    return f"{float(x):.16g}"


def ptag(x: float) -> str:
    return f"{float(x):g}".replace("-", "m").replace(".", "p")


def parse_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def linspace(a: float, b: float, n: int) -> np.ndarray:
    if n <= 1:
        return np.array([a], dtype=float)
    return np.linspace(a, b, n)


def trapz_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    w = np.empty_like(x)
    if len(x) == 1:
        w[0] = 0.0
        return w
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def omega_mesh(omega0: float, n_omega: int, T: float, x_thermal: float = 24.0) -> np.ndarray:
    if omega0 < 8.0 * T:
        raise ValueError(f"omega grid too narrow: need about +/-{8*T:g}, got +/-{omega0:g}")
    thermal_width = min(max(x_thermal * T, 5e-3), omega0)
    n_center = max(n_omega // 2, 101)
    n_outer = max(n_omega - n_center, 2)
    left = np.linspace(-omega0, -thermal_width, n_outer // 2, endpoint=False)
    center = np.linspace(-thermal_width, thermal_width, n_center)
    right = np.linspace(thermal_width, omega0, n_outer - len(left))
    return np.unique(np.sort(np.concatenate([left, center, right])))


def fermi(omega: np.ndarray, T: float) -> np.ndarray:
    x = omega / T
    out = np.empty_like(omega, dtype=float)
    out[x > 40.0] = 0.0
    out[x < -40.0] = 1.0
    mask = (x >= -40.0) & (x <= 40.0)
    out[mask] = 1.0 / (np.exp(x[mask]) + 1.0)
    return out


def fermi_window(omega: np.ndarray, T: float) -> np.ndarray:
    x = omega / (2.0 * T)
    out = np.zeros_like(omega, dtype=float)
    mask = np.abs(x) < 40.0
    out[mask] = 1.0 / (4.0 * T * np.cosh(x[mask]) ** 2)
    return out


def fk_impurity_green(hybrid, omega, eps_random, U, mu, w1, eta):
    z = omega + 1j * eta + mu - eps_random - hybrid
    return (1.0 - w1) / z + w1 / (z - U)


def disorder_averages(hybrid, omega, eps_grid, U, disorder_W, mu, w1, eta, block_size=256):
    rho_arith = np.zeros(len(omega), dtype=float)
    rho_typ = np.zeros(len(omega), dtype=float)
    eps_weights = trapz_weights(eps_grid)
    norm = disorder_W if len(eps_grid) > 1 else 1.0
    for start in range(0, len(omega), block_size):
        stop = min(start + block_size, len(omega))
        green_eps = fk_impurity_green(
            hybrid[start:stop, None],
            omega[start:stop, None],
            eps_grid[None, :],
            U,
            mu,
            w1,
            eta,
        )
        rho_eps = -np.imag(green_eps) / np.pi
        rho_arith[start:stop] = (rho_eps @ eps_weights) / norm
        rho_typ[start:stop] = np.exp((np.log(np.maximum(rho_eps, 1e-300)) @ eps_weights) / norm)
    return rho_arith, rho_typ


def green_from_ldos(omega, rho, eta, block_size=256):
    weights = trapz_weights(omega) * rho
    green = np.empty(len(omega), dtype=complex)
    imag = -np.pi * rho
    for start in range(0, len(omega), block_size):
        stop = min(start + block_size, len(omega))
        denom = omega[start:stop, None] - omega[None, :]
        kernel = np.divide(weights[None, :], denom, out=np.zeros_like(denom, dtype=float), where=np.abs(denom) > 0.0)
        real = np.sum(kernel, axis=1)
        green[start:stop] = real + 1j * imag[start:stop]
    return green


def filling_from_rho(omega, rho, T):
    return float(integrate_trapz(fermi(omega, T) * rho, omega))


def bethe_filling_zero_temperature(mu: float, band_halfwidth: float) -> float:
    D = float(band_halfwidth)
    if mu <= -D:
        return 0.0
    if mu >= D:
        return 1.0
    x = mu / D
    return 0.5 + (math.asin(x) + x * math.sqrt(max(1.0 - x * x, 0.0))) / math.pi


def bethe_mu_guess_for_filling(n: float, band_halfwidth: float) -> float:
    n = float(np.clip(n, 1e-8, 1.0 - 1e-8))
    lo = -float(band_halfwidth)
    hi = float(band_halfwidth)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bethe_filling_zero_temperature(mid, band_halfwidth) < n:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def bethe_green_noninteracting(omega: np.ndarray, mu: float, eta: float, band_halfwidth: float) -> np.ndarray:
    z = omega + 1j * eta + mu
    D = float(band_halfwidth)
    root = np.sqrt(z * z - D * D)
    root = np.where(np.real(z) < 0.0, -root, root)
    return 2.0 * (z - root) / (D * D)


def metallic_hybrid_guess(omega: np.ndarray, mu: float, eta: float, band_halfwidth: float) -> np.ndarray:
    t_bethe = 0.5 * band_halfwidth
    return (t_bethe * t_bethe) * bethe_green_noninteracting(omega, mu, eta, band_halfwidth)


def solve_dmft_for_mu(
    *,
    T: float,
    U: float,
    disorder_W: float,
    mu: float,
    w1: float,
    kind: str,
    omega0: float,
    n_omega: int,
    n_eps: int,
    eta: float,
    band_halfwidth: float,
    max_iter: int,
    tol: float,
    mix: float,
    initial_hybrid: np.ndarray | None = None,
) -> DMFTResult:
    t0 = time.time()
    omega = omega_mesh(omega0, n_omega, T)
    eps_grid = np.array([0.0]) if disorder_W <= 0.0 else np.linspace(-0.5 * disorder_W, 0.5 * disorder_W, n_eps)
    t_bethe = 0.5 * band_halfwidth
    if initial_hybrid is not None and len(initial_hybrid) == len(omega):
        hybrid = initial_hybrid.copy()
    else:
        hybrid = metallic_hybrid_guess(omega, mu, eta, band_halfwidth)
    diff = math.inf
    converged = False

    for iteration in range(1, max_iter + 1):
        rho_arith, rho_typ = disorder_averages(hybrid, omega, eps_grid, U, disorder_W, mu, w1, eta)
        rho_medium = rho_typ if kind == "typ" else rho_arith
        green_medium = green_from_ldos(omega, rho_medium, eta)
        hybrid_new = (t_bethe * t_bethe) * green_medium
        diff = float(np.max(np.abs(hybrid_new - hybrid)))
        hybrid = mix * hybrid_new + (1.0 - mix) * hybrid
        if diff < tol:
            converged = True
            break

    rho_arith, rho_typ = disorder_averages(hybrid, omega, eps_grid, U, disorder_W, mu, w1, eta)
    rho_medium = rho_typ if kind == "typ" else rho_arith
    green_medium = green_from_ldos(omega, rho_medium, eta)
    sigma = omega + 1j * eta + mu - hybrid - 1.0 / green_medium
    ne_actual = filling_from_rho(omega, rho_arith, T)
    return DMFTResult(
        omega=omega,
        rho_arith=rho_arith,
        rho_typ=rho_typ,
        rho_medium=rho_medium,
        green_medium=green_medium,
        sigma=sigma,
        hybrid=hybrid,
        mu=float(mu),
        ne_actual=ne_actual,
        iterations=iteration,
        converged=converged,
        diff=diff,
        runtime_sec=time.time() - t0,
    )


def solve_target_ne(
    *,
    ne_target: float,
    T: float,
    U: float,
    disorder_W: float,
    w1: float,
    kind: str,
    mu_guess: float,
    initial_hybrid: np.ndarray | None,
    mu_min: float,
    mu_max: float,
    ne_tol: float,
    max_mu_iter: int,
    accept_diff: float,
    previous_slope: float | None,
    **dmft_kwargs,
) -> tuple[DMFTResult, float | None]:
    history: list[tuple[float, float, DMFTResult]] = []

    def acceptable(f: float, res: DMFTResult) -> bool:
        return abs(f) <= ne_tol and res.diff <= accept_diff

    def eval_mu(mu: float, start_hybrid: np.ndarray | None) -> tuple[float, DMFTResult]:
        mu = float(np.clip(mu, mu_min, mu_max))
        res = solve_dmft_for_mu(
            T=T,
            U=U,
            disorder_W=disorder_W,
            mu=mu,
            w1=w1,
            kind=kind,
            initial_hybrid=start_hybrid,
            **dmft_kwargs,
        )
        f = res.ne_actual - ne_target
        history.append((res.mu, f, res))
        return f, res

    f0, r0 = eval_mu(mu_guess, initial_hybrid)
    if acceptable(f0, r0):
        return r0, previous_slope

    bracket = None
    if previous_slope is not None and abs(previous_slope) > 1e-10:
        mu1 = mu_guess - f0 / previous_slope
        f1, r1 = eval_mu(mu1, r0.hybrid)
        if acceptable(f1, r1):
            slope = (r1.ne_actual - r0.ne_actual) / (r1.mu - r0.mu) if r1.mu != r0.mu else previous_slope
            return r1, slope
        if f0 * f1 < 0.0:
            bracket = (r0.mu, f0, r0, r1.mu, f1, r1)

    if bracket is None:
        step = 0.25
        for _ in range(14):
            for mu2 in (mu_guess - step, mu_guess + step):
                f2, r2 = eval_mu(mu2, r0.hybrid)
                if acceptable(f2, r2):
                    slope = (r2.ne_actual - r0.ne_actual) / (r2.mu - r0.mu) if r2.mu != r0.mu else previous_slope
                    return r2, slope
                if f0 * f2 < 0.0:
                    bracket = (r0.mu, f0, r0, r2.mu, f2, r2)
                    break
            if bracket is not None:
                break
            step *= 2.0

    if bracket is None:
        flo, rlo = eval_mu(mu_min, r0.hybrid)
        fhi, rhi = eval_mu(mu_max, rlo.hybrid)
        if flo * fhi < 0.0:
            bracket = (rlo.mu, flo, rlo, rhi.mu, fhi, rhi)
        else:
            best = min(history, key=lambda item: abs(item[1]) + max(item[2].diff - accept_diff, 0.0) / max(accept_diff, 1e-12))
            return best[2], previous_slope

    lo, flo, rlo, hi, fhi, rhi = bracket
    def score(item: tuple[float, DMFTResult]) -> float:
        f, res = item
        return abs(f) + max(res.diff - accept_diff, 0.0) / max(accept_diff, 1e-12)

    best = rlo if score((flo, rlo)) < score((fhi, rhi)) else rhi
    slope = previous_slope
    for _ in range(max_mu_iter):
        if abs(fhi - flo) > 1e-14:
            mu_mid = hi - fhi * (hi - lo) / (fhi - flo)
            if not min(lo, hi) < mu_mid < max(lo, hi):
                mu_mid = 0.5 * (lo + hi)
        else:
            mu_mid = 0.5 * (lo + hi)
        fmid, rmid = eval_mu(mu_mid, best.hybrid)
        if score((fmid, rmid)) < score((best.ne_actual - ne_target, best)):
            best = rmid
        if acceptable(fmid, rmid):
            if rmid.mu != rlo.mu:
                slope = (rmid.ne_actual - rlo.ne_actual) / (rmid.mu - rlo.mu)
            return rmid, slope
        if flo * fmid < 0.0:
            hi, fhi, rhi = rmid.mu, fmid, rmid
        else:
            lo, flo, rlo = rmid.mu, fmid, rmid
    return best, slope


def solution_dir(base: Path, T: float, U: float, disorder_W: float, w1: float, kind: str) -> Path:
    return base / f"T_{ptag(T)}" / f"U_{ptag(U)}__Delta_{ptag(disorder_W)}__w1_{ptag(w1)}__{kind}"


def save_solution(path: Path, res: DMFTResult, meta: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path / "solution.npz",
        omega=res.omega,
        hybrid=res.hybrid,
        sigma=res.sigma,
        green_medium=res.green_medium,
        rho_arith=res.rho_arith,
        rho_typ=res.rho_typ,
        rho_medium=res.rho_medium,
    )
    with (path / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def load_solution(path: Path) -> tuple[dict, dict]:
    with (path / "metadata.json").open() as f:
        meta = json.load(f)
    data = np.load(path / "solution.npz")
    return meta, data


def bethe_transport_phi(eps, D):
    x = np.maximum(D * D - eps * eps, 0.0)
    return x**1.5


def sigma_causality_metrics(omega, sigma, T):
    im = np.imag(sigma)
    bad = im > 0.0
    thermal = np.abs(omega) < 10.0 * T
    bad_thermal = bad & thermal
    max_thermal = float(np.max(im[thermal])) if np.any(thermal) else float(np.max(im))
    return {
        "max_ImSigma_raw": float(np.max(im)),
        "max_ImSigma_thermal": max_thermal,
        "n_causality_bad": int(np.sum(bad)),
        "n_causality_bad_thermal": int(np.sum(bad_thermal)),
    }


def causalize_sigma_for_transport(sigma, eta):
    out = sigma.copy()
    bad = np.imag(out) > 0.0
    out[bad] = np.real(out[bad]) - 1j * eta
    return out


def transport_tau(omega, sigma, mu, n_band, band_halfwidth, eta, block_size=256):
    eps = np.linspace(-band_halfwidth, band_halfwidth, n_band + 2)[1:-1]
    phi = bethe_transport_phi(eps, band_halfwidth)
    eps_weights = trapz_weights(eps)
    tau = np.zeros_like(omega, dtype=float)
    for start in range(0, len(omega), block_size):
        stop = min(start + block_size, len(omega))
        z = omega[start:stop, None] + 1j * eta + mu - sigma[start:stop, None] - eps[None, :]
        a = -np.imag(1.0 / z) / np.pi
        tau[start:stop] = np.sum((a * a) * phi[None, :] * eps_weights[None, :], axis=1)
    return tau


def transport_from_arrays(omega, sigma, mu, T, n_band, band_halfwidth, eta):
    metrics = sigma_causality_metrics(omega, sigma, T)
    sigma_c = causalize_sigma_for_transport(sigma, eta)
    tau = transport_tau(omega, sigma_c, mu, n_band, band_halfwidth, eta)
    window = fermi_window(omega, T)
    L11 = float(integrate_trapz(tau * window, omega))
    L12 = float(integrate_trapz(omega * tau * window, omega))
    L22 = float(integrate_trapz(omega * omega * tau * window, omega))
    sigma_dc = L11
    kappa_bare = L22 / T
    correction = (L12 * L12 / L11) / T if abs(L11) > 1e-300 else np.nan
    kappa = kappa_bare - correction if np.isfinite(correction) else np.nan
    lorenz = kappa / (T * sigma_dc) if abs(sigma_dc) > 1e-300 else np.nan
    out = {
        "sigma": sigma_dc,
        "kappa": kappa,
        "kappa_bare": kappa_bare,
        "thermoelectric_correction": correction,
        "lorenz": lorenz,
        "L11": L11,
        "L12": L12,
        "L22": L22,
        "tau_min": float(np.nanmin(tau)),
        "tau_max": float(np.nanmax(tau)),
    }
    out.update(metrics)
    return out
