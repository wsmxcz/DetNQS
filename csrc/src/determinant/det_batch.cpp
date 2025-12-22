// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Determinant batch processing utilities.
 *
 * Provides efficient batch preparation of determinants for CI calculations:
 *   - Occupied orbital extraction
 *   - Excitation level (k) and phase computation
 *   - Hole-particle excitation lists with position mapping
 *
 * Phase calculation uses fermion anticommutation rules:
 *   phase = (-1)^(n_perm), where n_perm counts permutations to match
 *   canonical ordering during annihilation/creation operations.
 *
 * File: lever/determinant/det_batch.cpp
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: December, 2025
 */

#include <lever/determinant/det_batch.hpp>

#include <array>
#include <bit>
#include <stdexcept>

#include <lever/utils/bit_utils.hpp>

namespace lever::det_batch {

namespace {

constexpr int kMaxOrbU64 = 64;
constexpr int kMaxSO = 128;  // 2 * kMaxOrbU64

inline void check_norb(int n_orb) {
    if (n_orb < 0 || n_orb > kMaxOrbU64) {
        throw std::invalid_argument("n_orb out of range [0,64]");
    }
}

inline u64 norb_mask(int n_orb) {
    return make_mask<u64>(n_orb);
}

// Extract occupied orbital indices in ascending order
inline int fill_bits_asc(u64 word, int n_orb, int32_t* out, int max_count) {
    int n = 0;
    u64 m = word & norb_mask(n_orb);
    while (m) {
        const int idx = ctz(m);
        if (n < max_count) out[n] = static_cast<int32_t>(idx);
        ++n;
        m = clear_lsb(m);
    }
    return n;
}

// Compute phase factor for single spin channel via hole annihilation
// and particle creation sequence
inline int8_t phase_one_spin(u64 ref_word, u64 det_word, int n_orb) {
    const u64 m = norb_mask(n_orb);
    ref_word &= m;
    det_word &= m;

    u64 holes = ref_word & (~det_word);
    u64 parts = det_word & (~ref_word);

    if (popcount(holes) != popcount(parts)) {
        throw std::invalid_argument("holes != parts in spin channel");
    }

    u64 occ = ref_word;
    int8_t ph = 1;

    // Annihilate holes in descending order (MSB -> LSB)
    while (holes) {
        const int msb = 63 - clz(holes);
        const u64 bit = (u64{1} << static_cast<u64>(msb));
        const int nbelow = popcount_below<u64>(occ, msb);
        if (nbelow & 1) ph = static_cast<int8_t>(-ph);
        occ ^= bit;
        holes ^= bit;
    }

    // Create particles in ascending order (LSB -> MSB)
    while (parts) {
        const int lsb = ctz(parts);
        const u64 bit = (u64{1} << static_cast<u64>(lsb));
        const int nbelow = popcount_below<u64>(occ, lsb);
        if (nbelow & 1) ph = static_cast<int8_t>(-ph);
        occ ^= bit;
        parts = clear_lsb(parts);
    }

    return ph;
}

struct RefMaps {
    std::array<int32_t, kMaxSO> occ_pos{};
    std::array<int32_t, kMaxSO> virt_pos{};
};

// Build occupied and virtual orbital position maps for reference determinant
inline RefMaps build_ref_maps(Det ref, int n_orb, int n_alpha, int n_beta) {
    const int n_so = 2 * n_orb;
    const int n_e = n_alpha + n_beta;
    if (n_so > kMaxSO) {
        throw std::invalid_argument("n_so exceeds maximum");
    }

    RefMaps maps;
    maps.occ_pos.fill(-1);
    maps.virt_pos.fill(-1);

    std::array<int32_t, kMaxOrbU64> tmp_a{};
    std::array<int32_t, kMaxOrbU64> tmp_b{};

    const int na = fill_bits_asc(ref.alpha, n_orb, tmp_a.data(), kMaxOrbU64);
    const int nb = fill_bits_asc(ref.beta,  n_orb, tmp_b.data(), kMaxOrbU64);

    if (na != n_alpha || nb != n_beta) {
        throw std::invalid_argument("reference electron count mismatch");
    }

    // Occupied positions: alpha orbitals first, then beta (+n_orb offset)
    int pos = 0;
    for (int i = 0; i < n_alpha; ++i) {
        maps.occ_pos[tmp_a[i]] = static_cast<int32_t>(pos++);
    }
    for (int i = 0; i < n_beta; ++i) {
        maps.occ_pos[tmp_b[i] + n_orb] = static_cast<int32_t>(pos++);
    }

    // Virtual positions: complement of occupied in [0, 2*n_orb)
    int vpos = 0;
    for (int so = 0; so < n_so; ++so) {
        if (maps.occ_pos[so] < 0) {
            maps.virt_pos[so] = static_cast<int32_t>(vpos++);
        }
    }
    return maps;
}

// Fill excitation list in descending order with optional position mapping
inline void fill_desc_list(u64 mask, int n_orb, int offset,
                           int32_t* out_row, int kmax, int& cursor,
                           const RefMaps* maps, int32_t* pos_row, bool is_hole) {
    while (mask && cursor < kmax) {
        const int msb = 63 - clz(mask);
        const u64 bit = (u64{1} << static_cast<u64>(msb));
        const int so = msb + offset;

        out_row[cursor] = static_cast<int32_t>(so);

        if (maps && pos_row) {
            const int32_t p = is_hole ? maps->occ_pos[so] : maps->virt_pos[so];
            pos_row[cursor] = p;
        }

        ++cursor;
        mask ^= bit;
    }
}

// Fill excitation list in ascending order with optional position mapping
inline void fill_asc_list(u64 mask, int n_orb, int offset,
                          int32_t* out_row, int kmax, int& cursor,
                          const RefMaps* maps, int32_t* pos_row, bool is_hole) {
    while (mask && cursor < kmax) {
        const int lsb = ctz(mask);
        const u64 bit = (u64{1} << static_cast<u64>(lsb));
        const int so = lsb + offset;

        out_row[cursor] = static_cast<int32_t>(so);

        if (maps && pos_row) {
            const int32_t p = is_hole ? maps->occ_pos[so] : maps->virt_pos[so];
            pos_row[cursor] = p;
        }

        ++cursor;
        mask = clear_lsb(mask);
    }
}

}  // namespace

void prepare_det_batch(
    const u64* det_pairs,
    std::size_t B,
    Det ref,
    int n_orb,
    int n_alpha,
    int n_beta,
    const PrepareOptions& opt,
    int32_t* occ_out,
    int8_t*  k_out,
    int8_t*  phase_out,
    int32_t* holes_out,
    int32_t* parts_out,
    bool*    hp_mask_out,
    int32_t* holes_pos_out,
    int32_t* parts_pos_out
) {
    check_norb(n_orb);
    if (!det_pairs || !occ_out) {
        throw std::invalid_argument("null pointer in required outputs");
    }

    if (opt.need_hp_pos && !opt.need_hp) {
        throw std::invalid_argument("need_hp_pos requires need_hp");
    }
    if (opt.need_hp && opt.kmax < 0) {
        throw std::invalid_argument("kmax must be non-negative");
    }

    const int n_e = n_alpha + n_beta;
    const u64 m = norb_mask(n_orb);

    RefMaps maps_storage;
    const RefMaps* maps = nullptr;
    if (opt.need_hp_pos) {
        maps_storage = build_ref_maps(ref, n_orb, n_alpha, n_beta);
        maps = &maps_storage;
    }

    for (std::size_t i = 0; i < B; ++i) {
        const u64 a = det_pairs[2 * i + 0];
        const u64 b = det_pairs[2 * i + 1];

        if ((a & ~m) != 0 || (b & ~m) != 0) {
            throw std::invalid_argument("det bits outside n_orb range");
        }

        // Extract occupied orbitals: alpha (asc), then beta + n_orb (asc)
        int32_t* occ_row = occ_out + static_cast<std::ptrdiff_t>(i) * n_e;

        std::array<int32_t, kMaxOrbU64> tmpa{};
        std::array<int32_t, kMaxOrbU64> tmpb{};
        const int na = fill_bits_asc(a, n_orb, tmpa.data(), kMaxOrbU64);
        const int nb = fill_bits_asc(b, n_orb, tmpb.data(), kMaxOrbU64);

        if (na != n_alpha || nb != n_beta) {
            throw std::invalid_argument("electron count mismatch");
        }

        for (int j = 0; j < n_alpha; ++j) occ_row[j] = tmpa[j];
        for (int j = 0; j < n_beta;  ++j) occ_row[n_alpha + j] = tmpb[j] + n_orb;

        // Excitation level k = |holes_alpha| + |holes_beta|
        const u64 holes_a = (ref.alpha & (~a)) & m;
        const u64 holes_b = (ref.beta  & (~b)) & m;
        const int kk = popcount(holes_a) + popcount(holes_b);

        if (opt.need_k || opt.need_hp) {
            if (!k_out) throw std::invalid_argument("k_out is null");
            k_out[i] = static_cast<int8_t>(kk > 127 ? 127 : kk);
        }

        // Phase via fermion anticommutation
        if (opt.need_phase) {
            if (!phase_out) throw std::invalid_argument("phase_out is null");
            const int8_t pa = phase_one_spin(ref.alpha, a, n_orb);
            const int8_t pb = phase_one_spin(ref.beta,  b, n_orb);
            phase_out[i] = static_cast<int8_t>(pa * pb);
        }

        // Hole-particle excitation lists
        if (opt.need_hp) {
            const int kmax = opt.kmax;
            if (!holes_out || !parts_out || !hp_mask_out) {
                throw std::invalid_argument("hp outputs are null");
            }

            int32_t* holes_row = holes_out + static_cast<std::ptrdiff_t>(i) * kmax;
            int32_t* parts_row = parts_out + static_cast<std::ptrdiff_t>(i) * kmax;
            bool*    mask_row  = hp_mask_out + static_cast<std::ptrdiff_t>(i) * kmax;

            int32_t* holes_pos_row = nullptr;
            int32_t* parts_pos_row = nullptr;
            if (opt.need_hp_pos) {
                if (!holes_pos_out || !parts_pos_out) {
                    throw std::invalid_argument("pos outputs are null");
                }
                holes_pos_row = holes_pos_out + static_cast<std::ptrdiff_t>(i) * kmax;
                parts_pos_row = parts_pos_out + static_cast<std::ptrdiff_t>(i) * kmax;
            }

            // Initialize outputs
            for (int j = 0; j < kmax; ++j) {
                holes_row[j] = -1;
                parts_row[j] = -1;
                mask_row[j] = false;
                if (holes_pos_row) holes_pos_row[j] = -1;
                if (parts_pos_row) parts_pos_row[j] = -1;
            }

            const u64 parts_a = (a & (~ref.alpha)) & m;
            const u64 parts_b = (b & (~ref.beta )) & m;

            // Holes: alpha (desc), beta (desc) + n_orb
            int hcur = 0;
            fill_desc_list(holes_a, n_orb, 0,     holes_row, kmax, hcur, maps, holes_pos_row, true);
            fill_desc_list(holes_b, n_orb, n_orb, holes_row, kmax, hcur, maps, holes_pos_row, true);

            // Particles: alpha (asc), beta (asc) + n_orb
            int pcur = 0;
            fill_asc_list(parts_a, n_orb, 0,     parts_row, kmax, pcur, maps, parts_pos_row, false);
            fill_asc_list(parts_b, n_orb, n_orb, parts_row, kmax, pcur, maps, parts_pos_row, false);

            // Validity mask based on true k (capped at kmax)
            const int kcap = (kk < 0) ? 0 : (kk > kmax ? kmax : kk);
            for (int j = 0; j < kmax; ++j) {
                mask_row[j] = (j < kcap);
            }
        }
    }
}

}  // namespace lever::det_batch
