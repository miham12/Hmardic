
from __future__ import annotations

import math
import time
from typing import Optional, Tuple, List

import numpy as np

# Optional acceleration: numba (massive speed-up on long sequences).
try:
    from numba import njit  # type: ignore
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False
    njit = None  # type: ignore

# Numerical floor for probabilities / rates before taking log.
# User requested 1e-100.
_LOG_EPS = 1e-100

# ──────────────────────────────────────────────────────────────────────────────
# Global log-factorial cache: LOG_FACT[k] = log(k!)
# Shared across all PoissonHMM instances *within a process*.
# (Each worker process has its own address space, so call precompute_log_fact()
#  in each worker initializer.)
# ──────────────────────────────────────────────────────────────────────────────

_LOG_FACT_CACHE = np.array([0.0], dtype=np.float64)  # log(0!) = 0


def precompute_log_fact(max_n: int = 10000) -> None:
    """Precompute global log-factorials up to max_n (inclusive).

    Safe to call multiple times; will only grow the cache if needed.
    """
    _ensure_log_fact_cache(int(max_n))


def _ensure_log_fact_cache(mx: int) -> None:
    """Ensure global cache covers 0..mx inclusive."""
    global _LOG_FACT_CACHE
    if mx <= 0:
        return
    cur = int(_LOG_FACT_CACHE.shape[0]) - 1
    if mx <= cur:
        return

    # Extend with vectorized log + cumsum for speed.
    # We know: log_fact[k] = log_fact[cur] + sum_{i=cur+1..k} log(i)
    start = cur + 1
    # Make new array
    new = np.empty(mx + 1, dtype=np.float64)
    new[: cur + 1] = _LOG_FACT_CACHE
    tail = np.log(np.arange(start, mx + 1, dtype=np.float64))
    new[start:] = new[cur] + np.cumsum(tail)
    _LOG_FACT_CACHE = new


def _logaddexp2(a: float, b: float) -> float:
    return float(np.logaddexp(a, b))


def _logaddexp4(a: float, b: float, c: float, d: float) -> float:
    return float(np.logaddexp(np.logaddexp(a, b), np.logaddexp(c, d)))


if _HAVE_NUMBA:
    @njit(cache=True)
    def _forward_nb(emiss: np.ndarray, logA: np.ndarray, logpi: np.ndarray) -> np.ndarray:
        T = emiss.shape[0]
        la = np.empty((T, 2), dtype=np.float64)
        la[0, 0] = logpi[0] + emiss[0, 0]
        la[0, 1] = logpi[1] + emiss[0, 1]
        for t in range(1, T):
            prev0 = la[t - 1, 0]
            prev1 = la[t - 1, 1]
            a0 = prev0 + logA[0, 0]
            a1 = prev1 + logA[1, 0]
            la[t, 0] = np.logaddexp(a0, a1) + emiss[t, 0]
            b0 = prev0 + logA[0, 1]
            b1 = prev1 + logA[1, 1]
            la[t, 1] = np.logaddexp(b0, b1) + emiss[t, 1]
        return la

    @njit(cache=True)
    def _backward_nb(emiss: np.ndarray, logA: np.ndarray) -> np.ndarray:
        T = emiss.shape[0]
        lb = np.empty((T, 2), dtype=np.float64)
        lb[T - 1, 0] = 0.0
        lb[T - 1, 1] = 0.0
        for t in range(T - 2, -1, -1):
            v00 = logA[0, 0] + emiss[t + 1, 0] + lb[t + 1, 0]
            v01 = logA[0, 1] + emiss[t + 1, 1] + lb[t + 1, 1]
            lb[t, 0] = np.logaddexp(v00, v01)
            v10 = logA[1, 0] + emiss[t + 1, 0] + lb[t + 1, 0]
            v11 = logA[1, 1] + emiss[t + 1, 1] + lb[t + 1, 1]
            lb[t, 1] = np.logaddexp(v10, v11)
        return lb

    @njit(cache=True)
    def _xi_sum_nb(la: np.ndarray, lb: np.ndarray, emiss: np.ndarray, logA: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return xi_sum (2x2) across t and loglik."""
        T = la.shape[0]
        loglik = np.logaddexp(la[T - 1, 0], la[T - 1, 1])

        xi00 = 0.0
        xi01 = 0.0
        xi10 = 0.0
        xi11 = 0.0

        for t in range(T - 1):
            p00 = la[t, 0] + logA[0, 0] + emiss[t + 1, 0] + lb[t + 1, 0]
            p01 = la[t, 0] + logA[0, 1] + emiss[t + 1, 1] + lb[t + 1, 1]
            p10 = la[t, 1] + logA[1, 0] + emiss[t + 1, 0] + lb[t + 1, 0]
            p11 = la[t, 1] + logA[1, 1] + emiss[t + 1, 1] + lb[t + 1, 1]
            z = np.logaddexp(np.logaddexp(p00, p01), np.logaddexp(p10, p11))
            xi00 += math.exp(p00 - z)
            xi01 += math.exp(p01 - z)
            xi10 += math.exp(p10 - z)
            xi11 += math.exp(p11 - z)

        xi_sum = np.empty((2, 2), dtype=np.float64)
        xi_sum[0, 0] = xi00
        xi_sum[0, 1] = xi01
        xi_sum[1, 0] = xi10
        xi_sum[1, 1] = xi11
        return xi_sum, float(loglik)

    @njit(cache=True)
    def _viterbi_nb(emiss: np.ndarray, logA: np.ndarray, logpi: np.ndarray) -> np.ndarray:
        T = emiss.shape[0]
        states = np.empty(T, dtype=np.int8)
        if T == 0:
            return states

        d0 = np.empty(T, dtype=np.float64)
        d1 = np.empty(T, dtype=np.float64)
        psi0 = np.empty(T, dtype=np.int8)
        psi1 = np.empty(T, dtype=np.int8)

        d0[0] = logpi[0] + emiss[0, 0]
        d1[0] = logpi[1] + emiss[0, 1]
        psi0[0] = 0
        psi1[0] = 0

        for t in range(1, T):
            v00 = d0[t - 1] + logA[0, 0]
            v10 = d1[t - 1] + logA[1, 0]
            if v00 >= v10:
                psi0[t] = 0
                d0[t] = v00 + emiss[t, 0]
            else:
                psi0[t] = 1
                d0[t] = v10 + emiss[t, 0]

            v01 = d0[t - 1] + logA[0, 1]
            v11 = d1[t - 1] + logA[1, 1]
            if v01 >= v11:
                psi1[t] = 0
                d1[t] = v01 + emiss[t, 1]
            else:
                psi1[t] = 1
                d1[t] = v11 + emiss[t, 1]

        states[T - 1] = 0 if d0[T - 1] >= d1[T - 1] else 1
        for t in range(T - 2, -1, -1):
            if states[t + 1] == 0:
                states[t] = psi0[t + 1]
            else:
                states[t] = psi1[t + 1]
        return states


class PoissonHMM:
    """Fast 2-state Poisson HMM for integer counts with per-bin baseline lambdas.

    Emissions:
      state 0 (background):  count[t] ~ Poisson(p_bg  * lambda[t])
      state 1 (peak):        count[t] ~ Poisson(alpha * lambda[t])
    """

    def __init__(
        self,
        counts: np.ndarray,
        lambdas: np.ndarray,
        *,
        A: Optional[np.ndarray] = None,
        pi: Optional[np.ndarray] = None,
        p_bg: float = 1.0,
        alpha: float = 5.0,
        max_iter: int = 12,
        tol: float = 1e-2,
        assume_preprocessed: bool = False,
    ):
        """assume_preprocessed=True means:
        - counts is int64, non-negative, 1D
        - lambdas is float64, 1D, same length
        This avoids extra conversions/checks for speed (pipeline already guarantees this).
        """
        if assume_preprocessed:
            c = counts
            lam = lambdas
        else:
            c = np.asarray(counts)
            if c.dtype.kind == "f":
                c = np.floor(c).astype(np.int64, copy=False)
            else:
                c = c.astype(np.int64, copy=False)
            if c.ndim != 1:
                raise ValueError("counts must be a 1D array")
            if np.any(c < 0):
                raise ValueError("counts must be non-negative")

            lam = np.asarray(lambdas, dtype=np.float64)
            if lam.ndim != 1:
                raise ValueError("lambdas must be a 1D array")
            if lam.shape[0] != c.shape[0]:
                raise ValueError("counts and lambdas must have same length")

        self.counts_i = c
        self.lambdas = lam
        self.T = int(c.shape[0])

        self.p_bg = float(p_bg)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.A = (
            np.array([[0.99, 0.01], [0.95, 0.05]], dtype=np.float64)
            if A is None
            else np.asarray(A, dtype=np.float64)
        )
        self.pi = (
            np.array([0.99, 0.01], dtype=np.float64)
            if pi is None
            else np.asarray(pi, dtype=np.float64)
        )

        # Ensure cache covers this sequence (cheap if already precomputed)
        if self.T:
            _ensure_log_fact_cache(int(self.counts_i.max()))

    @property
    def have_numba(self) -> bool:
        return _HAVE_NUMBA

    def _emiss_log(self) -> np.ndarray:
        """Return (T,2) emission log-probabilities."""
        T = self.T
        if T == 0:
            return np.empty((0, 2), dtype=np.float64)

        n = self.counts_i
        mx = int(n.max())
        if mx >= _LOG_FACT_CACHE.shape[0]:
            _ensure_log_fact_cache(mx)

        lam = np.clip(self.lambdas, _LOG_EPS, None)

        mu0 = np.clip(self.p_bg * lam, _LOG_EPS, None)
        mu1 = np.clip(self.alpha * lam, _LOG_EPS, None)

        lf = _LOG_FACT_CACHE[n]  # vectorized indexing
        nf = n.astype(np.float64, copy=False)
        emiss0 = nf * np.log(mu0) - mu0 - lf
        emiss1 = nf * np.log(mu1) - mu1 - lf

        out = np.empty((T, 2), dtype=np.float64)
        out[:, 0] = emiss0
        out[:, 1] = emiss1
        return out

    def _forward(self, emiss: np.ndarray) -> np.ndarray:
        if self.T == 0:
            return np.empty((0, 2), dtype=np.float64)
        logA = np.log(np.clip(self.A, _LOG_EPS, None))
        logpi = np.log(np.clip(self.pi, _LOG_EPS, None))
        if _HAVE_NUMBA:
            return _forward_nb(emiss, logA, logpi)

        T = self.T
        la = np.empty((T, 2), dtype=np.float64)
        la[0, 0] = logpi[0] + emiss[0, 0]
        la[0, 1] = logpi[1] + emiss[0, 1]
        for t in range(1, T):
            prev0, prev1 = la[t - 1, 0], la[t - 1, 1]
            la[t, 0] = _logaddexp2(prev0 + logA[0, 0], prev1 + logA[1, 0]) + emiss[t, 0]
            la[t, 1] = _logaddexp2(prev0 + logA[0, 1], prev1 + logA[1, 1]) + emiss[t, 1]
        return la

    def _backward(self, emiss: np.ndarray) -> np.ndarray:
        if self.T == 0:
            return np.empty((0, 2), dtype=np.float64)
        logA = np.log(np.clip(self.A, _LOG_EPS, None))
        if _HAVE_NUMBA:
            return _backward_nb(emiss, logA)

        T = self.T
        lb = np.empty((T, 2), dtype=np.float64)
        lb[T - 1, 0] = 0.0
        lb[T - 1, 1] = 0.0
        for t in range(T - 2, -1, -1):
            v00 = logA[0, 0] + emiss[t + 1, 0] + lb[t + 1, 0]
            v01 = logA[0, 1] + emiss[t + 1, 1] + lb[t + 1, 1]
            lb[t, 0] = _logaddexp2(v00, v01)

            v10 = logA[1, 0] + emiss[t + 1, 0] + lb[t + 1, 0]
            v11 = logA[1, 1] + emiss[t + 1, 1] + lb[t + 1, 1]
            lb[t, 1] = _logaddexp2(v10, v11)
        return lb

    def _gamma_and_xi_sum(
        self, log_alpha: np.ndarray, log_beta: np.ndarray, emiss: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return gamma (T,2), xi_sum (2,2), loglik."""
        T = self.T
        if T == 0:
            return np.empty((0, 2), dtype=np.float64), np.empty((2, 2), dtype=np.float64), float("-inf")

        # gamma: posterior P(z_t=i | x) ∝ alpha_t(i) * beta_t(i)
        log_gamma = log_alpha + log_beta
        norm = np.logaddexp(log_gamma[:, 0], log_gamma[:, 1])
        gamma = np.empty_like(log_gamma)
        gamma[:, 0] = np.exp(log_gamma[:, 0] - norm)
        gamma[:, 1] = np.exp(log_gamma[:, 1] - norm)

        logA = np.log(np.clip(self.A, _LOG_EPS, None))

        if T == 1:
            loglik = float(np.logaddexp(log_alpha[0, 0], log_alpha[0, 1]))
            return gamma, np.zeros((2, 2), dtype=np.float64), loglik

        if _HAVE_NUMBA:
            xi_sum, loglik = _xi_sum_nb(log_alpha, log_beta, emiss, logA)
            return gamma, xi_sum, loglik

        loglik = float(np.logaddexp(log_alpha[T - 1, 0], log_alpha[T - 1, 1]))
        xi_sum = np.zeros((2, 2), dtype=np.float64)
        for t in range(T - 1):
            p00 = log_alpha[t, 0] + logA[0, 0] + emiss[t + 1, 0] + log_beta[t + 1, 0]
            p01 = log_alpha[t, 0] + logA[0, 1] + emiss[t + 1, 1] + log_beta[t + 1, 1]
            p10 = log_alpha[t, 1] + logA[1, 0] + emiss[t + 1, 0] + log_beta[t + 1, 0]
            p11 = log_alpha[t, 1] + logA[1, 1] + emiss[t + 1, 1] + log_beta[t + 1, 1]
            z = _logaddexp4(p00, p01, p10, p11)
            xi_sum[0, 0] += math.exp(p00 - z)
            xi_sum[0, 1] += math.exp(p01 - z)
            xi_sum[1, 0] += math.exp(p10 - z)
            xi_sum[1, 1] += math.exp(p11 - z)
        return gamma, xi_sum, loglik

    def _gamma_and_loglik(self, log_alpha: np.ndarray, log_beta: np.ndarray) -> Tuple[np.ndarray, float]:
        T = self.T
        if T == 0:
            return np.empty((0, 2), dtype=np.float64), float("-inf")

        log_gamma = log_alpha + log_beta
        norm = np.logaddexp(log_gamma[:, 0], log_gamma[:, 1])
        gamma = np.empty_like(log_gamma)
        gamma[:, 0] = np.exp(log_gamma[:, 0] - norm)
        gamma[:, 1] = np.exp(log_gamma[:, 1] - norm)

        loglik = float(np.logaddexp(log_alpha[T - 1, 0], log_alpha[T - 1, 1]))
        return gamma, loglik

    def baum_welch_train(
        self,
        *,
        verbose: bool = False,
        log_every: int = 1,
        prefix: str = "",
        profile_time: bool = False,
        update_transitions: bool = True,
        update_pi: bool = True,
    ) -> float:
        prev = float("-inf")

        for it in range(self.max_iter):
            t_it0 = time.perf_counter()

            t0 = time.perf_counter()
            emiss = self._emiss_log()
            t_emiss = time.perf_counter() - t0

            t0 = time.perf_counter()
            la = self._forward(emiss)
            t_fwd = time.perf_counter() - t0

            t0 = time.perf_counter()
            lb = self._backward(emiss)
            t_bwd = time.perf_counter() - t0

            t0 = time.perf_counter()
            if update_transitions:
                gamma, xi_sum, loglik = self._gamma_and_xi_sum(la, lb, emiss)
            else:
                gamma, loglik = self._gamma_and_loglik(la, lb)
                xi_sum = None
            t_gx = time.perf_counter() - t0

            t0 = time.perf_counter()
            if update_pi:
                g0 = gamma[0]
                s0 = float(g0.sum())
                if s0 > 0:
                    self.pi = g0 / s0

            if update_transitions and xi_sum is not None:
                row_sums = xi_sum.sum(axis=1, keepdims=True)
                self.A = np.divide(xi_sum, row_sums, out=np.full_like(xi_sum, 0.5), where=row_sums > 0)

            lam = np.clip(self.lambdas, _LOG_EPS, None)
            # counts_i is int; dot will upcast to float internally
            num0 = float(np.dot(gamma[:, 0], self.counts_i))
            den0 = float(np.dot(gamma[:, 0], lam))
            num1 = float(np.dot(gamma[:, 1], self.counts_i))
            den1 = float(np.dot(gamma[:, 1], lam))
            if den0 > 0:
                self.p_bg = num0 / den0
            if den1 > 0:
                self.alpha = num1 / den1

            if self.alpha < self.p_bg:
                self.alpha, self.p_bg = self.p_bg, self.alpha
                if update_pi:
                    self.pi = self.pi[::-1].copy()
                if update_transitions:
                    self.A = self.A[::-1, ::-1].copy()
            t_m = time.perf_counter() - t0

            t_it = time.perf_counter() - t_it0
            if verbose and (it % max(int(log_every), 1) == 0):
                tag = f"{prefix} " if prefix else ""
                if profile_time:
                    print(
                        f"[BW] {tag}iter={it:02d} loglik={loglik:.3f} Δ={loglik-prev:+.3f} "
                        f"p_bg={self.p_bg:.4g} alpha={self.alpha:.4g} "
                        f"A=[[{self.A[0,0]:.4f},{self.A[0,1]:.4f}],[{self.A[1,0]:.4f},{self.A[1,1]:.4f}]] "
                        f"pi=[{self.pi[0]:.4f},{self.pi[1]:.4f}] "
                        f"time={t_it:.3f}s (emiss {t_emiss:.3f}s, fwd {t_fwd:.3f}s, bwd {t_bwd:.3f}s, "
                        f"gamma/xi {t_gx:.3f}s, mstep {t_m:.3f}s) "
                        f"numba={int(_HAVE_NUMBA)}"
                    )
                else:
                    print(
                        f"[BW] {tag}iter={it:02d} loglik={loglik:.3f} Δ={loglik-prev:+.3f} "
                        f"p_bg={self.p_bg:.4g} alpha={self.alpha:.4g}"
                    )

            if abs(loglik - prev) < self.tol:
                prev = loglik
                break
            prev = loglik
        return prev

    @classmethod
    def fit_multi(
        cls,
        counts_list: List[np.ndarray],
        lambdas_list: List[np.ndarray],
        *,
        p_bg: float = 1.0,
        alpha: float = 5.0,
        A: Optional[np.ndarray] = None,
        pi: Optional[np.ndarray] = None,
        max_iter: int = 30,
        tol: float = 1e-5,
        verbose: bool = False,
        log_every: int = 1,
        prefix: str = "",
        profile_time: bool = False,
        update_transitions: bool = True,
        update_pi: bool = True,
        assume_preprocessed: bool = True,
    ) -> "PoissonHMM":
        if len(counts_list) != len(lambdas_list):
            raise ValueError("counts_list and lambdas_list must have same length")

        first_idx = None
        for i, c in enumerate(counts_list):
            if c is not None and len(c) > 0:
                first_idx = i
                break
        if first_idx is None:
            return cls(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
                p_bg=p_bg,
                alpha=alpha,
                A=A,
                pi=pi,
                max_iter=max_iter,
                tol=tol,
                assume_preprocessed=True,
            )

        model = cls(
            counts=counts_list[first_idx],
            lambdas=lambdas_list[first_idx],
            p_bg=p_bg,
            alpha=alpha,
            A=A,
            pi=pi,
            max_iter=max_iter,
            tol=tol,
            assume_preprocessed=assume_preprocessed,
        )

        prev = float("-inf")
        for it in range(model.max_iter):
            t_it0 = time.perf_counter()

            pi_acc = np.zeros(2, dtype=np.float64)
            xi_sum_acc = np.zeros((2, 2), dtype=np.float64) if update_transitions else None
            num0 = den0 = num1 = den1 = 0.0
            loglik_total = 0.0

            t_emiss = t_fwd = t_bwd = t_gx = t_m = 0.0

            for c, lam in zip(counts_list, lambdas_list):
                if c is None or len(c) == 0:
                    continue

                if assume_preprocessed:
                    model.counts_i = c
                    model.lambdas = lam
                else:
                    cc = np.asarray(c)
                    if cc.dtype.kind == "f":
                        cc = np.floor(cc).astype(np.int64, copy=False)
                    else:
                        cc = cc.astype(np.int64, copy=False)
                    cc[cc < 0] = 0
                    model.counts_i = cc
                    model.lambdas = np.asarray(lam, dtype=np.float64)

                model.T = int(model.counts_i.shape[0])

                # Ensure global log_fact cache (cheap if already precomputed)
                if model.T:
                    _ensure_log_fact_cache(int(model.counts_i.max()))

                t0 = time.perf_counter()
                emiss = model._emiss_log()
                t_emiss += time.perf_counter() - t0

                t0 = time.perf_counter()
                la = model._forward(emiss)
                t_fwd += time.perf_counter() - t0

                t0 = time.perf_counter()
                lb = model._backward(emiss)
                t_bwd += time.perf_counter() - t0

                t0 = time.perf_counter()
                if update_transitions:
                    gamma, xi_sum, loglik = model._gamma_and_xi_sum(la, lb, emiss)
                else:
                    gamma, loglik = model._gamma_and_loglik(la, lb)
                    xi_sum = None
                t_gx += time.perf_counter() - t0

                loglik_total += float(loglik)
                if update_pi:
                    pi_acc += gamma[0]
                if update_transitions and xi_sum_acc is not None and xi_sum is not None:
                    xi_sum_acc += xi_sum

                lam_clip = np.clip(model.lambdas, _LOG_EPS, None)
                num0 += float(np.dot(gamma[:, 0], model.counts_i))
                den0 += float(np.dot(gamma[:, 0], lam_clip))
                num1 += float(np.dot(gamma[:, 1], model.counts_i))
                den1 += float(np.dot(gamma[:, 1], lam_clip))

            t0 = time.perf_counter()
            if update_pi:
                s0 = float(pi_acc.sum())
                if s0 > 0:
                    model.pi = pi_acc / s0

            if update_transitions and xi_sum_acc is not None:
                row_sums = xi_sum_acc.sum(axis=1, keepdims=True)
                model.A = np.divide(xi_sum_acc, row_sums, out=np.full_like(xi_sum_acc, 0.5), where=row_sums > 0)

            if den0 > 0:
                model.p_bg = num0 / den0
            if den1 > 0:
                model.alpha = num1 / den1

            if model.alpha < model.p_bg:
                model.alpha, model.p_bg = model.p_bg, model.alpha
                if update_pi:
                    model.pi = model.pi[::-1].copy()
                if update_transitions:
                    model.A = model.A[::-1, ::-1].copy()
            t_m += time.perf_counter() - t0

            t_it = time.perf_counter() - t_it0
            if verbose and (it % max(int(log_every), 1) == 0):
                tag = f"{prefix} " if prefix else ""
                if profile_time:
                    print(
                        f"[BW-multi] {tag}iter={it:02d} loglik={loglik_total:.3f} Δ={loglik_total-prev:+.3f} "
                        f"p_bg={model.p_bg:.4g} alpha={model.alpha:.4g} "
                        f"A=[[{model.A[0,0]:.4f},{model.A[0,1]:.4f}],[{model.A[1,0]:.4f},{model.A[1,1]:.4f}]] "
                        f"pi=[{model.pi[0]:.4f},{model.pi[1]:.4f}] "
                        f"time={t_it:.3f}s (emiss {t_emiss:.3f}s, fwd {t_fwd:.3f}s, bwd {t_bwd:.3f}s, "
                        f"gamma/xi {t_gx:.3f}s, mstep {t_m:.3f}s) "
                        f"numba={int(_HAVE_NUMBA)}"
                    )
                else:
                    print(
                        f"[BW-multi] {tag}iter={it:02d} loglik={loglik_total:.3f} Δ={loglik_total-prev:+.3f} "
                        f"p_bg={model.p_bg:.4g} alpha={model.alpha:.4g}"
                    )

            if abs(loglik_total - prev) < model.tol:
                prev = loglik_total
                break
            prev = loglik_total

        return model

    def decode_peak(self) -> Tuple[np.ndarray, np.ndarray]:
        emiss = self._emiss_log()
        states = self._viterbi_from_emiss(emiss)
        la = self._forward(emiss)
        lb = self._backward(emiss)
        p_peak, _loglik = self._posterior_peak_and_loglik(la, lb)
        return states, p_peak

    @staticmethod
    def _posterior_peak_and_loglik(log_alpha: np.ndarray, log_beta: np.ndarray) -> Tuple[np.ndarray, float]:
        if log_alpha.shape[0] == 0:
            return np.empty((0,), dtype=np.float64), float("-inf")
        log0 = log_alpha[:, 0] + log_beta[:, 0]
        log1 = log_alpha[:, 1] + log_beta[:, 1]
        norm = np.logaddexp(log0, log1)
        p1 = np.exp(log1 - norm).astype(np.float64, copy=False)
        loglik = float(np.logaddexp(log_alpha[-1, 0], log_alpha[-1, 1]))
        return p1, loglik

    def _viterbi_from_emiss(self, emiss: np.ndarray) -> np.ndarray:
        logA = np.log(np.clip(self.A, _LOG_EPS, None))
        logpi = np.log(np.clip(self.pi, _LOG_EPS, None))
        if _HAVE_NUMBA:
            return _viterbi_nb(emiss, logA, logpi)
        return self._viterbi_py(emiss, logA, logpi)

    @staticmethod
    def _viterbi_py(emiss: np.ndarray, logA: np.ndarray, logpi: np.ndarray) -> np.ndarray:
        T = emiss.shape[0]
        if T == 0:
            return np.zeros(0, dtype=np.int8)

        d0 = np.empty(T, dtype=np.float64)
        d1 = np.empty(T, dtype=np.float64)
        psi = np.empty((T, 2), dtype=np.int8)

        d0[0] = logpi[0] + emiss[0, 0]
        d1[0] = logpi[1] + emiss[0, 1]
        psi[0, 0] = 0
        psi[0, 1] = 0

        for t in range(1, T):
            v00 = d0[t - 1] + logA[0, 0]
            v10 = d1[t - 1] + logA[1, 0]
            if v00 >= v10:
                psi[t, 0] = 0
                d0[t] = v00 + emiss[t, 0]
            else:
                psi[t, 0] = 1
                d0[t] = v10 + emiss[t, 0]

            v01 = d0[t - 1] + logA[0, 1]
            v11 = d1[t - 1] + logA[1, 1]
            if v01 >= v11:
                psi[t, 1] = 0
                d1[t] = v01 + emiss[t, 1]
            else:
                psi[t, 1] = 1
                d1[t] = v11 + emiss[t, 1]

        states = np.empty(T, dtype=np.int8)
        states[T - 1] = 0 if d0[T - 1] >= d1[T - 1] else 1
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states
