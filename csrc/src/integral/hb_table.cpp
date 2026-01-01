// Copyright 2025 The Nebula-QC Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file hb_table.cpp
 * @brief Heat-bath coupling table for FCIQMC excitation generation.
 *
 * Stores pre-screened virtual pair couplings |⟨ij||ab⟩| for efficient
 * stochastic sampling. Uses CSR-like layout with descending weight ordering
 * within each row for cutoff-based rejection sampling.
 *
 * Algorithm: For each occupied pair (i,j), enumerate all spin-conserving
 * virtual pairs (a,b) and store weights w = |⟨ij||ab⟩| above threshold.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#include <detnqs/integral/hb_table.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace detnqs {

HeatBathTable::HeatBathTable(const IntegralSO& so_ints, HBBuildOptions opt)
    : so_(so_ints)
    , n_so_(so_ints.get_n_so())
    , num_rows_(0)
    , opt_(opt)
{
    if (n_so_ < 2) {
        throw std::invalid_argument("HeatBathTable requires n_so >= 2");
    }
    // Row count = unordered pairs {i,j} where i>j
    num_rows_ = static_cast<std::size_t>(n_so_) * (n_so_ - 1) / 2;
}

void HeatBathTable::build() {
    std::vector<std::vector<Entry>> rows(num_rows_);
    std::size_t nnz_total = 0;
    
    std::vector<Entry> scratch;
    scratch.reserve(256);  // Heuristic capacity for reuse

    // Enumerate all occupied pairs (i,j) with i>j
    for (int i = 1; i < n_so_; ++i) {
        for (int j = 0; j < i; ++j) {
            build_row_(i, j, scratch, nnz_total);
            const std::size_t rid = unordered_pair_index(i, j);
            rows[rid] = std::move(scratch);
            scratch.clear();
        }
    }

    finalize_layout_(rows, nnz_total);
}

std::size_t HeatBathTable::row_id(int i, int j) const noexcept {
    if (i == j) return std::numeric_limits<std::size_t>::max();
    return unordered_pair_index(i, j);
}

std::size_t HeatBathTable::row_size(int i, int j) const noexcept {
    const auto rid = row_id(i, j);
    if (rid == std::numeric_limits<std::size_t>::max()) return 0;
    return row_offsets_[rid + 1] - row_offsets_[rid];
}

double HeatBathTable::row_weight_sum(int i, int j) const noexcept {
    const auto rid = row_id(i, j);
    if (rid == std::numeric_limits<std::size_t>::max()) return 0.0;
    return row_sum_[rid];
}

HeatBathTable::RowView HeatBathTable::row_view(int i, int j) const noexcept {
    RowView view{};
    const auto rid = row_id(i, j);
    if (rid == std::numeric_limits<std::size_t>::max()) return view;

    const std::size_t off = row_offsets_[rid];
    const std::size_t len = row_offsets_[rid + 1] - off;
    
    view.len = len;
    if (len == 0) return view;

    view.a = &a_[off];
    view.b = &b_[off];
    view.w = &w_[off];
    
    return view;
}

std::size_t HeatBathTable::memory_bytes() const noexcept {
    return row_offsets_.size() * sizeof(std::size_t)
         + a_.size() * sizeof(int)
         + b_.size() * sizeof(int)
         + w_.size() * sizeof(double)
         + row_sum_.size() * sizeof(double);
}

void HeatBathTable::build_row_(int i, int j, std::vector<Entry>& buffer, 
                                std::size_t& nnz_total) {
    const double thresh = opt_.threshold;
    
    // Spin pattern: even index = α, odd index = β
    const int s_i = (i & 1);
    const int s_j = (j & 1);

    // Enumerate virtual pairs (a,b) with a>b, disjoint from {i,j}
    for (int a = 1; a < n_so_; ++a) {
        for (int b = 0; b < a; ++b) {
            if (a == i || a == j || b == i || b == j) continue;

            // Spin conservation: (s_a, s_b) must match (s_i, s_j) pattern
            const int s_a = (a & 1);
            const int s_b = (b & 1);
            const bool spin_ok = (s_a == s_i && s_b == s_j) || 
                                  (s_a == s_j && s_b == s_i);
            if (!spin_ok) continue;

            // Weight = |⟨ij||ab⟩| with antisymmetrization
            const double val = so_.get_h2e_anti(i, j, a, b);
            const double weight = std::abs(val);
            
            if (weight >= thresh) {
                buffer.push_back({a, b, weight});
            }
        }
    }

    if (buffer.empty()) return;

    // Sort descending by weight (enables binary search in cutoff sampling)
    std::ranges::sort(buffer, std::greater{}, &Entry::w);

    nnz_total += buffer.size();
}

void HeatBathTable::finalize_layout_(const std::vector<std::vector<Entry>>& rows, 
                                      std::size_t nnz_total) {
    // Build CSR row offsets
    row_offsets_.assign(num_rows_ + 1, 0);
    std::size_t cursor = 0;
    for (std::size_t rid = 0; rid < num_rows_; ++rid) {
        row_offsets_[rid] = cursor;
        cursor += rows[rid].size();
    }
    row_offsets_[num_rows_] = cursor;

    // Flatten into SoA (Structure of Arrays) layout
    a_.resize(nnz_total);
    b_.resize(nnz_total);
    w_.resize(nnz_total);
    row_sum_.assign(num_rows_, 0.0);

    for (std::size_t rid = 0; rid < num_rows_; ++rid) {
        const std::size_t off = row_offsets_[rid];
        const auto& row = rows[rid];
        
        for (std::size_t k = 0; k < row.size(); ++k) {
            a_[off + k] = row[k].a;
            b_[off + k] = row[k].b;
            w_[off + k] = row[k].w;
            row_sum_[rid] += row[k].w;
        }
    }
}

} // namespace detnqs
