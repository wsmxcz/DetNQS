// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file det_ops.cpp
 * @brief Implementation of determinant operations (analysis & collectors).
 */

#include <lever/determinant/det_ops.hpp>
#include <lever/utils/bit_utils.hpp>

#include <limits>

namespace lever::det_ops {

// ============================================================================
// Internal helpers
// ============================================================================

namespace {

struct SpinDiff {
    std::array<int, MAX_EXCITATION_DEGREE> holes{};
    std::array<int, MAX_EXCITATION_DEGREE> parts{};
    int n_exc = 0; // 0,1,2 or >2
};

/**
 * @brief Extract holes (ket&~bra) and particles (bra&~ket) indices for one spin.
 * Indices are filled in ascending order (LSB->MSB).
 */
[[nodiscard]] SpinDiff extract_spin_diff(u64 bra_occ, u64 ket_occ) noexcept {
    SpinDiff d;

    const u64 holes_mask = ket_occ & ~bra_occ; // holes: ket\bra
    const u64 parts_mask = bra_occ & ~ket_occ; // parts: bra\ket

#ifndef NDEBUG
    // Electron number must be conserved within each spin for valid <bra|ket>.
    assert(popcount(holes_mask) == popcount(parts_mask));
#endif
    const int nh = popcount(holes_mask);
    const int np = popcount(parts_mask);
    if (nh != np) { d.n_exc = MAX_EXCITATION_DEGREE + 1; return d; }

    d.n_exc = nh;
    if (d.n_exc > MAX_EXCITATION_DEGREE) return d;

    int k = 0;
    for (u64 x = holes_mask; x; x = clear_lsb(x)) d.holes[k++] = ctz(x);
    k = 0;
    for (u64 x = parts_mask; x; x = clear_lsb(x)) d.parts[k++] = ctz(x);
    return d;
}

/**
 * @brief Count occupied orbitals strictly between positions (lo, hi).
 * Uses the occupation of the *bra* determinant (Algorithm 4).
 * Safe bitmasking via make_mask avoids undefined shifts at 64.
 */
[[nodiscard]] inline int count_between(u64 bra_occ, int a, int b) noexcept {
    if (a == b) return 0;
    const int lo = (a < b) ? a : b;
    const int hi = (a < b) ? b : a;
    const u64 mask = make_mask<u64>(hi) & ~make_mask<u64>(lo + 1); // (lo, hi)
    return popcount(bra_occ & mask);
}

/**
 * @brief Parity for one spin channel per Scemama & Giner (Algorithm 4).
 * Sum over pairs: #(occupied between hole and particle); if 2 excitations,
 * add +1 when the two intervals cross.
 */
[[nodiscard]] inline int parity_one_spin(u64 bra_occ,
                                         int n_exc,
                                         const std::array<int, MAX_EXCITATION_DEGREE>& holes,
                                         const std::array<int, MAX_EXCITATION_DEGREE>& parts) noexcept {
    int nperm = 0;
    for (int i = 0; i < n_exc; ++i) {
        nperm += count_between(bra_occ, holes[i], parts[i]);
    }
    if (n_exc == 2) {
        const int a = std::min(holes[0], parts[0]);
        const int b = std::max(holes[0], parts[0]);
        const int c = std::min(holes[1], parts[1]);
        const int d = std::max(holes[1], parts[1]);
        const bool crossed = (c > a && c < b && d > b) || (a > c && a < d && b > d);
        if (crossed) ++nperm;
    }
    return nperm; // caller takes mod 2
}

/** Binomial for reserve hints (saturates on overflow). */
[[nodiscard]] size_t n_choose_k(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n / 2) k = n - k;
    size_t r = 1;
    for (int i = 1; i <= k; ++i) {
        const size_t num = static_cast<size_t>(n - i + 1);
        // Guard overflow before multiply
        if (r > std::numeric_limits<size_t>::max() / num) {
            return std::numeric_limits<size_t>::max() / 2; // saturate
        }
        r *= num;
        r /= static_cast<size_t>(i);
    }
    return r;
}

} // namespace

// ============================================================================
// Excitation analysis & phase
// ============================================================================

ExcInfo analyze_excitation(const Det& bra, const Det& ket) noexcept {
    ExcInfo info;

    // Per-spin extraction with unified holes/particles convention
    const SpinDiff a = extract_spin_diff(bra.alpha, ket.alpha);
    const SpinDiff b = extract_spin_diff(bra.beta , ket.beta );

    info.n_alpha_exc = static_cast<u8>(a.n_exc);
    info.n_beta_exc  = static_cast<u8>(b.n_exc);
    info.degree      = static_cast<u8>(info.n_alpha_exc + info.n_beta_exc);

    if (info.degree > MAX_EXCITATION_DEGREE) {
        info.phase = 0.0; // irrelevant; element is zero by Slater–Condon
        return info;
    }

    // Copy indices (already ascending within each list)
    for (int i = 0; i < a.n_exc; ++i) { info.holes_alpha[i] = a.holes[i]; info.particles_alpha[i] = a.parts[i]; }
    for (int i = 0; i < b.n_exc; ++i) { info.holes_beta [i] = b.holes[i]; info.particles_beta [i] = b.parts[i]; }

    // Phase (per spin; no cross-spin term)
    int nperm = 0;
    nperm += parity_one_spin(bra.alpha, a.n_exc, info.holes_alpha, info.particles_alpha);
    nperm += parity_one_spin(bra.beta , b.n_exc, info.holes_beta , info.particles_beta );
    info.phase = (nperm & 1) ? -1.0 : 1.0;

    return info;
}

f64 phase(const Det& bra, const Det& ket) noexcept {
    // Recompute minimal data to avoid copying ExcInfo when only the sign is needed.
    const SpinDiff a = extract_spin_diff(bra.alpha, ket.alpha);
    const SpinDiff b = extract_spin_diff(bra.beta , ket.beta );
    if (a.n_exc + b.n_exc > MAX_EXCITATION_DEGREE) return 0.0; // undefined/zero element

    int nperm = 0;
    nperm += parity_one_spin(bra.alpha, a.n_exc, a.holes, a.parts);
    nperm += parity_one_spin(bra.beta , b.n_exc, b.holes, b.parts);
    return (nperm & 1) ? -1.0 : 1.0;
}

// ============================================================================
// Batch collections
// ============================================================================

std::vector<Det> collect_singles(const Det& ket, int n_orb) {
    std::vector<Det> out;
    const size_t na = popcount(ket.alpha), nb = popcount(ket.beta);
    const size_t va = (n_orb > static_cast<int>(na)) ? (n_orb - static_cast<int>(na)) : 0;
    const size_t vb = (n_orb > static_cast<int>(nb)) ? (n_orb - static_cast<int>(nb)) : 0;
    out.reserve(na * va + nb * vb);

    for_each_single(ket, n_orb, [&](const Det& d){ out.push_back(d); });
    return out;
}

std::vector<Det> collect_doubles(const Det& ket, int n_orb) {
    std::vector<Det> out;
    const int na = popcount(ket.alpha), nb = popcount(ket.beta);
    const int va = n_orb - na, vb = n_orb - nb;
    const size_t daa = n_choose_k(na, 2) * n_choose_k(std::max(va,0), 2);
    const size_t dbb = n_choose_k(nb, 2) * n_choose_k(std::max(vb,0), 2);
    const size_t dab = static_cast<size_t>(std::max(na,0) * std::max(va,0)) *
                       static_cast<size_t>(std::max(nb,0) * std::max(vb,0));
    out.reserve(daa + dbb + dab);

    for_each_double(ket, n_orb, [&](const Det& d){ out.push_back(d); });
    return out;
}

std::vector<Det> collect_connected(const Det& ket, int n_orb) {
    std::vector<Det> out;
    const int na = popcount(ket.alpha), nb = popcount(ket.beta);
    const int va = n_orb - na, vb = n_orb - nb;
    const size_t singles = static_cast<size_t>(std::max(na,0) * std::max(va,0) + std::max(nb,0) * std::max(vb,0));
    const size_t daa = n_choose_k(na, 2) * n_choose_k(std::max(va,0), 2);
    const size_t dbb = n_choose_k(nb, 2) * n_choose_k(std::max(vb,0), 2);
    const size_t dab = static_cast<size_t>(std::max(na,0) * std::max(va,0)) *
                       static_cast<size_t>(std::max(nb,0) * std::max(vb,0));
    out.reserve(singles + daa + dbb + dab);

    for_each_connected(ket, n_orb, [&](const Det& d){ out.push_back(d); });
    return out;
}

} // namespace lever::det_ops