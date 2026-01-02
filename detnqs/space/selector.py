# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Determinant selection strategies based on importance scores.

Streaming support:
  - If stream=True, selectors accept `scores` as an iterable/factory yielding:
        (scores_block: np.ndarray [B], dets_block: np.ndarray [B,2])
  - For multi-pass selectors (Threshold/TopFraction), `scores` MUST be a callable
    factory that returns a fresh iterator each call.
"""

from __future__ import annotations

from typing import Protocol, Any, Callable
import heapq
import math
import itertools

import numpy as np


def _blocks_factory(scores: Any) -> Callable[[], Any]:
    """
    Normalize input into a re-iterable factory.
    - If `scores` is callable: assumed to be a factory returning a fresh iterator.
    - Else: wrap by lambda: iter(scores). (Only safe for single-pass usage.)
    """
    if callable(scores):
        return scores
    return lambda: iter(scores)


def _logsumexp_1d(a: np.ndarray) -> float:
    """Stable logsumexp for 1D array."""
    if a.size == 0:
        return -np.inf
    m = float(np.max(a))
    return m + float(np.log(np.sum(np.exp(a - m))))


class Selector(Protocol):
    """Protocol for determinant selection strategies."""

    def select(self, scores: Any, dets: Any) -> np.ndarray:
        """
        Args:
            scores:
              - np.ndarray [N] when stream=False
              - iterable/factory yielding (scores_blk [B], dets_blk [B,2]) when stream=True
            dets:
              - np.ndarray [N,2] when stream=False
              - ignored (can be None) when stream=True
        """
        ...


class TopKSelector:
    """Select k determinants with highest scores."""

    def __init__(self, k: int, stream: bool = False) -> None:
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
            idx = np.argpartition(scores, -k)[-k:]
            idx = idx[np.argsort(scores[idx])[::-1]]
            return dets[idx]

        # Streaming path: keep a min-heap of size k for top-k largest scores.
        k = max(0, int(self.k))
        if k == 0:
            return np.zeros((0, 2), dtype=np.uint64)

        blocks = _blocks_factory(scores)
        heap: list[tuple[float, int, np.ndarray]] = []
        counter = itertools.count()  # tie-breaker to avoid comparing np.ndarray

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

        # Sort by score descending (ignore counter for ranking).
        heap.sort(key=lambda x: x[0], reverse=True)
        return np.stack([d for _, _, d in heap], axis=0)


class TopFractionSelector:
    """Select top fraction of determinants by cumulative probability mass."""

    def __init__(
        self,
        fraction: float,
        min_k: int = 1,
        max_k: int | None = None,
        stream: bool = False,
    ) -> None:
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

        # Streaming TopFraction needs bounded candidate pool.
        if self.max_k is None:
            raise ValueError("TopFractionSelector(stream=True) requires max_k to bound memory.")

        frac = float(self.fraction)
        if not (0.0 < frac <= 1.0):
            return np.zeros((0, 2), dtype=np.uint64)

        blocks = _blocks_factory(scores)

        # Pass 1: global logZ = log sum_i exp(2*log|psi_i|)
        logZ = -np.inf
        for s_blk, _ in blocks():
            s_blk = np.asarray(s_blk, dtype=np.float64)
            logZ = np.logaddexp(logZ, _logsumexp_1d(2.0 * s_blk))

        target_log_mass = math.log(frac)

        # Adaptive candidate pool size M (starts small, doubles until enough mass or hits max_k).
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

            # Check if candidate pool captures enough global probability mass.
            log_mass_cand = _logsumexp_1d(logp_cand)
            if log_mass_cand < target_log_mass and M < int(self.max_k):
                M = min(int(self.max_k), M * 2)
                continue

            # Sort by logp desc and take minimal k reaching fraction.
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
        self.threshold = float(threshold)
        self.max_size = int(max_size) if max_size is not None else None
        self.stream = bool(stream)

    def select(self, scores: Any, dets: Any) -> np.ndarray:
        if not self.stream:
            dets = np.asarray(dets, dtype=np.uint64)
            if dets.size == 0:
                return dets
            log_amp = np.asarray(scores, dtype=np.float64)

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

            local_prob = prob[passed_idx]
            k = self.max_size
            idx_local = np.argpartition(local_prob, -k)[-k:]
            idx_local = idx_local[np.argsort(local_prob[idx_local])[::-1]]
            return dets[passed_idx[idx_local]]

        thr = float(self.threshold)
        if thr <= 0.0:
            return np.zeros((0, 2), dtype=np.uint64)

        blocks = _blocks_factory(scores)

        # Pass 1: global logZ
        logZ = -np.inf
        for s_blk, _ in blocks():
            s_blk = np.asarray(s_blk, dtype=np.float64)
            logZ = np.logaddexp(logZ, _logsumexp_1d(2.0 * s_blk))

        log_thr = math.log(thr)

        kcap = self.max_size
        if kcap is not None and kcap > 0:
            # Keep best kcap by logp using heap with tie-breaker.
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

        # No cap: may be huge if threshold too small (user responsibility).
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