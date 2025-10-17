// Copyright 2025 The Nebula-QC Authors
// SPDX-License-Identifier: Apache-2.0

#include <lever/integral/hb_table.hpp>
#include <cmath>

namespace lever {

HeatBathTable::HeatBathTable(const IntegralSO& so_ints, HBBuildOptions opt)
    : so_(so_ints)
    , n_so_(so_ints.get_n_so())
    , num_rows_(0)
    , opt_(opt)
{
    if (n_so_ < 2) {
        throw std::invalid_argument("HeatBathTable: n_so must be >= 2.");
    }
    // Number of unordered pairs {i>j}
    num_rows_ = static_cast<std::size_t>(n_so_) * (n_so_ - 1) / 2;
}

void HeatBathTable::build() {
    // Temporary storage for each row
    std::vector<std::vector<Entry>> rows(num_rows_);
    std::size_t nnz_total = 0;
    
    // Reusable buffer for single-row construction
    std::vector<Entry> scratch;
    scratch.reserve(256);  // Heuristic initial capacity

    // Build all rows
    for (int i = 1; i < n_so_; ++i) {
        for (int j = 0; j < i; ++j) {
            build_row_(i, j, scratch, nnz_total);
            const std::size_t rid = unordered_pair_index(i, j);
            rows[rid] = std::move(scratch);
            scratch.clear();
        }
    }

    // Finalize CSR-like flat layout
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
    
    // Spin pattern of occupied pair (even=alpha, odd=beta)
    const int si = (i & 1);
    const int sj = (j & 1);

    // Enumerate all virtual pairs (a>b) disjoint from {i,j}
    for (int a = 1; a < n_so_; ++a) {
        for (int b = 0; b < a; ++b) {
            // Skip if overlaps with occupied
            if (a == i || a == j || b == i || b == j) continue;

            // Spin-preserving check: (a,b) must match (i,j) pattern
            const int sa = (a & 1);
            const int sb = (b & 1);
            const bool pattern_ok = (sa == si && sb == sj) || (sa == sj && sb == si);
            if (!pattern_ok) continue;

            // Compute antisymmetrized integral
            const double val = so_.get_h2e_anti(i, j, a, b);
            const double weight = std::abs(val);
            
            if (weight >= thresh) {
                buffer.push_back({a, b, weight});
            }
        }
    }

    if (buffer.empty()) return;

    // Sort descending by weight (critical for with_cutoff() binary search)
    std::sort(buffer.begin(), buffer.end(), 
              [](const Entry& x, const Entry& y) { 
                  return x.w > y.w; 
              });

    nnz_total += buffer.size();
}

void HeatBathTable::finalize_layout_(const std::vector<std::vector<Entry>>& rows, 
                                      std::size_t nnz_total) {
    // Build CSR offsets
    row_offsets_.assign(num_rows_ + 1, 0);
    std::size_t cursor = 0;
    for (std::size_t rid = 0; rid < num_rows_; ++rid) {
        row_offsets_[rid] = cursor;
        cursor += rows[rid].size();
    }
    row_offsets_[num_rows_] = cursor;

    // Flatten into contiguous arrays
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

} // namespace lever
