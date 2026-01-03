# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Determinant selection strategies for variational set refinement.

Provides selectors (TopK, TopFraction, Threshold) to choose the next
variational set V_{k+1} from target set T_k based on importance scores.

Streaming mode:
  - When stream=True, scores can be an iterable/factory yielding blocks:
    (scores_chunk: ndarray [B], dets_chunk: ndarray [B,2])
  - Multi-pass selectors (TopFraction, Threshold) require scores as
    a callable factory returning fresh iterators.

File: lever/selectors.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from typing import Protocol, Any, Callable
import heapq
import math
import itertools

import numpy as np


def _blocks_factory(scores: Any) -> Callable[[], Any]:
    """Normalize input into a re-iterable factory."""
    if callable(scores):
        return scores
    return lambda: iter(scores)


def _logsumexp_1d(a: np.ndarray) -> float:
    """Numerically stable log(sum_i exp(a_i)) for 1D array."""
    if a.size == 0:
        return -np.inf
    m = float(np.max(a))
    return m + float(np.log(np.sum(np.exp(a - m))))


class Selector(Protocol):
    """Protocol for determinant selection strategies."""

    def select(self, scores: Any, dets: Any) -> np.ndarray:
        """
        Select determinants from target set T_k for next variational set V_{k+1}.

        Args:
            scores: Importance scores (log amplitudes). When stream=False, ndarray [N].
                   When stream=True, iterable/factory yielding (scores_chunk, dets_chunk).
            dets: Determinant bitstrings [N,2] when stream=False; ignored when stream=True.

        Returns:
            Selected determinants as ndarray [K,2].
        """
        ...


class TopKSelector:
    """Select K determinants with highest importance scores."""

    def __init__(self, k: int, stream: bool = False) -> None:
        """
        Args:
            k: Number of determinants to select.
            stream: Enable streaming mode for large datasets.
        """
        self.k = int(k)
        self.stream = bool(stream)

    def select(self, scores: Any, dets: Any) -> np.ndarray:
        if not self.stream:
            dets = np.asarray(dets, dtype=np.uint64)
            if dets.size == 0:
                return dets
            scores = np.asarray(scores, dtype=np.float64)
            n = scores.shape[0]
            k = min(self.k, n)
            if k <= 0:
                return dets[:0]
          
            # Partial sort: O(n + k log k) instead of O(n log n)
            idx = np.argpartition(scores, -k)[-k:]
            idx = idx[np.argsort(scores[idx])[::-1]]
            return dets[idx]

        # Streaming: maintain a min-heap of top-k entries
        k = max(0, int(self.k))
        if k == 0:
            return np.zeros((0, 2), dtype=np.uint64)

        blocks = _blocks_factory(scores)
        heap: list[tuple[float, int, np.ndarray]] = []
        counter = itertools.count()  # Tie-breaker for stable sorting

        for s_blk, d_blk in blocks():
            if d_blk is None or len(d_blk) == 0:
                continue
            s_blk = np.asarray(s_blk, dtype=np.float64)
            d_blk = np.asarray(d_blk, dtype=np.uint64)

            kk = min(k, int(s_blk.shape[0]))
            idx = np.argpartition(s_blk, -kk)[-kk:]

            for s, d in zip(s_blk[idx], d_blk[idx]):
                item = (float(s), next(counter), d.copy())
                if len(heap) < k:
                    heapq.heappush(heap, item)
                else:
                    heapq.heappushpop(heap, item)

        if not heap:
            return np.zeros((0, 2), dtype=np.uint64)

        heap.sort(key=lambda x: x[0], reverse=True)
        return np.stack([d for _, _, d in heap], axis=0)


class TopFractionSelector:
    """
    Select determinants capturing a target fraction of total probability mass.

    Uses normalized probabilities p_i = |psi_i|^2 / Z with Z = sum_j |psi_j|^2.
    Selects minimal K such that sum_{i=1..K} p_i >= fraction.
    """

    def __init__(
        self,
        fraction: float,
        min_k: int = 1,
        max_k: int | None = None,
        stream: bool = False,
    ) -> None:
        """
        Args:
            fraction: Target cumulative probability (0 < fraction <= 1).
            min_k: Minimum number of determinants to select.
            max_k: Maximum allowed selections (required for stream=True).
            stream: Enable streaming mode.
        """
        self.fraction = float(fraction)
        self.min_k = int(min_k)
        self.max_k = int(max_k) if max_k is not None else None
        self.stream = bool(stream)

    def select(self, scores: Any, dets: Any) -> np.ndarray:
        if not self.stream:
            dets = np.asarray(dets, dtype=np.uint64)
            if dets.size == 0:
                return dets
            log_amp = np.asarray(scores, dtype=np.float64)

            # Compute normalized probabilities: p_i = |psi_i|^2 / sum_j |psi_j|^2
            log_prob = 2.0 * log_amp
            log_prob_max = np.max(log_prob)
            prob_unnorm = np.exp(log_prob - log_prob_max)
            prob = prob_unnorm / np.sum(prob_unnorm)

            idx = np.argsort(prob)[::-1]
            cdf = np.cumsum(prob[idx])

            k = int(np.searchsorted(cdf, self.fraction, side="left")) + 1
            n = log_amp.shape[0]
            k = max(k, self.min_k)
            if self.max_k is not None:
                k = min(k, self.max_k)
            k = min(k, n)
            if k <= 0:
                return dets[:0]
            return dets[idx[:k]]

        if self.max_k is None:
            raise ValueError("TopFractionSelector(stream=True) requires max_k to bound memory.")

        frac = float(self.fraction)
        if not (0.0 < frac <= 1.0):
            return np.zeros((0, 2), dtype=np.uint64)

        blocks = _blocks_factory(scores)

        # Pass 1: compute global log Z = log(sum_i exp(2*log|psi_i|))
        logZ = -np.inf
        for s_blk, _ in blocks():
            s_blk = np.asarray(s_blk, dtype=np.float64)
            logZ = np.logaddexp(logZ, _logsumexp_1d(2.0 * s_blk))

        target_log_mass = math.log(frac)

        # Adaptive candidate pool: start small, double until sufficient mass or max_k
        M = min(int(self.max_k), max(int(self.min_k), 4096))

        while True:
            heap: list[tuple[float, int, np.ndarray]] = []
            counter = itertools.count()

            for s_blk, d_blk in blocks():
                s_blk = np.asarray(s_blk, dtype=np.float64)
                d_blk = np.asarray(d_blk, dtype=np.uint64)
                logp_blk = 2.0 * s_blk - logZ

                kk = min(M, int(logp_blk.shape[0]))
                idx = np.argpartition(logp_blk, -kk)[-kk:]
                for lp, d in zip(logp_blk[idx], d_blk[idx]):
                    item = (float(lp), next(counter), d.copy())
                    if len(heap) < M:
                        heapq.heappush(heap, item)
                    else:
                        heapq.heappushpop(heap, item)

            if not heap:
                return np.zeros((0, 2), dtype=np.uint64)

            logp_cand = np.array([lp for lp, _, _ in heap], dtype=np.float64)
            dets_cand = np.stack([d for _, _, d in heap], axis=0)

            # Check if candidate pool captures enough probability mass
            log_mass_cand = _logsumexp_1d(logp_cand)
            if log_mass_cand < target_log_mass and M < int(self.max_k):
                M = min(int(self.max_k), M * 2)
                continue

            # Find minimal k reaching target fraction
            order = np.argsort(logp_cand)[::-1]
            log_cum = -np.inf
            k_need = 0
            for j, idx in enumerate(order):
                log_cum = np.logaddexp(log_cum, logp_cand[idx])
                if log_cum >= target_log_mass:
                    k_need = j + 1
                    break
            if k_need == 0:
                k_need = int(self.max_k)

            k = max(k_need, int(self.min_k))
            k = min(k, int(self.max_k))
            return dets_cand[order[:k]]


class ThresholdSelector:
    """Select determinants with normalized probability above threshold."""

    def __init__(self, threshold: float, max_size: int | None = None, stream: bool = False) -> None:
        """
        Args:
            threshold: Minimum normalized probability p_i = |psi_i|^2 / Z.
            max_size: Maximum selections (keeps top-max_size if exceeded).
            stream: Enable streaming mode.
        """
        self.threshold = float(threshold)
        self.max_size = int(max_size) if max_size is not None else None
        self.stream = bool(stream)

    def select(self, scores: Any, dets: Any) -> np.ndarray:
        if not self.stream:
            dets = np.asarray(dets, dtype=np.uint64)
            if dets.size == 0:
                return dets
            log_amp = np.asarray(scores, dtype=np.float64)

            # Compute normalized probabilities
            log_prob = 2.0 * log_amp
            log_prob_max = np.max(log_prob)
            prob_unnorm = np.exp(log_prob - log_prob_max)
            prob = prob_unnorm / np.sum(prob_unnorm)

            mask = prob >= self.threshold
            if not np.any(mask):
                return dets[:0]

            passed_idx = np.nonzero(mask)[0]
            if self.max_size is None or passed_idx.size <= self.max_size:
                return dets[passed_idx]

            # Truncate to max_size keeping highest probabilities
            local_prob = prob[passed_idx]
            k = self.max_size
            idx_local = np.argpartition(local_prob, -k)[-k:]
            idx_local = idx_local[np.argsort(local_prob[idx_local])[::-1]]
            return dets[passed_idx[idx_local]]

        thr = float(self.threshold)
        if thr <= 0.0:
            return np.zeros((0, 2), dtype=np.uint64)

        blocks = _blocks_factory(scores)

        # Pass 1: compute global log Z
        logZ = -np.inf
        for s_blk, _ in blocks():
            s_blk = np.asarray(s_blk, dtype=np.float64)
            logZ = np.logaddexp(logZ, _logsumexp_1d(2.0 * s_blk))

        log_thr = math.log(thr)

        kcap = self.max_size
        if kcap is not None and kcap > 0:
            # Maintain top-kcap by log probability
            heap: list[tuple[float, int, np.ndarray]] = []
            counter = itertools.count()

            for s_blk, d_blk in blocks():
                s_blk = np.asarray(s_blk, dtype=np.float64)
                d_blk = np.asarray(d_blk, dtype=np.uint64)
                logp = 2.0 * s_blk - logZ

                mask = logp >= log_thr
                if not np.any(mask):
                    continue

                for lp, d in zip(logp[mask], d_blk[mask]):
                    item = (float(lp), next(counter), d.copy())
                    if len(heap) < kcap:
                        heapq.heappush(heap, item)
                    else:
                        heapq.heappushpop(heap, item)

            if not heap:
                return np.zeros((0, 2), dtype=np.uint64)

            heap.sort(key=lambda x: x[0], reverse=True)
            return np.stack([d for _, _, d in heap], axis=0)

        # No cap: collect all passing determinants
        out: list[np.ndarray] = []
        for s_blk, d_blk in blocks():
            s_blk = np.asarray(s_blk, dtype=np.float64)
            d_blk = np.asarray(d_blk, dtype=np.uint64)
            logp = 2.0 * s_blk - logZ
            mask = logp >= log_thr
            if np.any(mask):
                out.append(d_blk[mask])

        if not out:
            return np.zeros((0, 2), dtype=np.uint64)
        return np.concatenate(out, axis=0)


__all__ = ["Selector", "TopKSelector", "TopFractionSelector", "ThresholdSelector"]
