// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file integral_mo.cpp
 * @brief Molecular orbital integral storage with FCIDUMP parser.
 *
 * Stores 1e (h_pq) and 2e (⟨pq|rs⟩) integrals in triangular packed format.
 * Parses FCIDUMP files with chemists' notation: ⟨pq|rs⟩ = ∫∫ φ_p(1)φ_q(1) r₁₂⁻¹ φ_r(2)φ_s(2).
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#include <detnqs/integral/integral_mo.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace detnqs {

IntegralMO::IntegralMO(int num_orbitals) : n_orbs(num_orbitals) {
    if (num_orbitals <= 0 || num_orbitals > 64) {
        throw std::invalid_argument("Number of orbitals must be in [1, 64]");
    }
    
    // Triangular packing: n(n+1)/2 for symmetric matrices
    const size_t n1e = static_cast<size_t>(n_orbs) * (n_orbs + 1) / 2;
    const size_t n2e = n1e * (n1e + 1) / 2;
    
    h1e_.resize(n1e, 0.0);
    h2e_.resize(n2e, 0.0);
}

void IntegralMO::load_from_fcidump(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open FCIDUMP: " + filename);
    }

    parse_namelist(file);
    
    // Rewind to parse integral data
    file.clear();
    file.seekg(0, std::ios::beg);
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || is_comment_line(line) || line.find('&') != std::string::npos) {
            continue;
        }
        parse_integral_line(line);
    }
}

std::pair<size_t, size_t> IntegralMO::get_nonzero_count(double threshold) const noexcept {
    auto count_nonzero = [threshold](const auto& vec) {
        return std::ranges::count_if(vec, [threshold](double v) {
            return std::abs(v) > threshold;
        });
    };
    
    return {count_nonzero(h1e_), count_nonzero(h2e_)};
}

void IntegralMO::parse_namelist(std::ifstream& file) {
    std::string line, namelist;
    bool in_namelist = false;
    
    while (std::getline(file, line)) {
        if (!in_namelist && (line.find("&FCI") != std::string::npos || 
                             line.find("&fci") != std::string::npos)) {
            in_namelist = true;
        }
        
        if (in_namelist) {
            namelist += line + " ";
            if (line.find("&END") != std::string::npos || 
                line.find("&end") != std::string::npos) {
                break;
            }
        }
    }
    
    if (!in_namelist) {
        throw std::runtime_error("No &FCI namelist found in FCIDUMP");
    }
    
    // Verify NORB matches constructor argument
    int file_norb = 0;
    if (extract_parameter(namelist, "NORB", file_norb) && file_norb != n_orbs) {
        throw std::runtime_error("NORB mismatch: expected " + std::to_string(n_orbs) + 
                                 ", got " + std::to_string(file_norb));
    }
    
    if (!extract_parameter(namelist, "NELEC", n_elecs) || n_elecs < 0) {
        throw std::runtime_error("Invalid or missing NELEC in namelist");
    }
    
    int ms2 = 0;
    if (extract_parameter(namelist, "MS2", ms2)) {
        spin_mult = std::abs(ms2) + 1;
    }
}

bool IntegralMO::extract_parameter(const std::string& content, 
                                   const std::string& param, 
                                   int& value) {
    auto to_upper = [](std::string s) {
        std::ranges::transform(s, s.begin(), ::toupper);
        return s;
    };
    
    const auto upper_content = to_upper(content);
    const auto upper_param = to_upper(param);
    const auto key = upper_param + "=";
    
    auto pos = upper_content.find(key);
    if (pos == std::string::npos) {
        return false;
    }
    
    pos += key.length();
    
    // Skip leading whitespace
    while (pos < content.length() && std::isspace(content[pos])) {
        ++pos;
    }
    
    // Extract integer value
    size_t end_pos = pos;
    while (end_pos < content.length() && 
           (std::isdigit(content[end_pos]) || content[end_pos] == '-' || content[end_pos] == '+')) {
        ++end_pos;
    }
    
    if (end_pos > pos) {
        value = std::stoi(content.substr(pos, end_pos - pos));
        return true;
    }
    
    return false;
}

bool IntegralMO::is_comment_line(const std::string& line) noexcept {
    if (line.empty()) {
        return true;
    }
    
    constexpr char comment_chars[] = {'!', '*', 'C', 'c', '#'};
    return std::ranges::any_of(comment_chars, [&](char c) { return line[0] == c; });
}

void IntegralMO::parse_integral_line(std::string& line) {
    // Convert Fortran 'D' exponent to C++ 'E' format
    std::ranges::transform(line, line.begin(), [](char c) {
        return (c == 'D' || c == 'd') ? 'E' : c;
    });
    
    double value;
    int p, q, r, s;
    std::istringstream iss(line);

    if (!(iss >> value >> p >> q >> r >> s)) {
        return;
    }
    
    // FCIDUMP format (1-based indexing):
    //   E_nuc:       value  0  0  0  0
    //   h_pq:        value  p  q  0  0
    //   ⟨pq|rs⟩:     value  p  q  r  s  (chemists' notation)
    if (p == 0 && q == 0 && r == 0 && s == 0) {
        e_nuc = value;
    } else if (r == 0 && s == 0) {
        h1e_[h1e_key(p - 1, q - 1)] = value;  // Convert to 0-based indexing
    } else {
        h2e_[h2e_key(p - 1, q - 1, r - 1, s - 1)] = value;  // Convert to 0-based indexing
    }
}

} // namespace detnqs
