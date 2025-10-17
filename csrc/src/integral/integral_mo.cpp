// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file integral_mo.cpp
 * @brief Implementation of molecular orbital integrals management.
 * @author Zheng (Alex) Che, email: wsmxcz@gmail.com
 * @date July, 2025
 */
#include <lever/integral/integral_mo.hpp>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cmath>

namespace lever {

IntegralMO::IntegralMO(int num_orbitals) : n_orbs(num_orbitals) {
    if (num_orbitals <= 0 || num_orbitals > 64) {
        throw std::invalid_argument("IntegralMO: Number of orbitals must be between 1 and 64.");
    }
    
    // Calculate storage sizes for triangular packing
    const size_t n1e_elements = static_cast<size_t>(n_orbs) * (n_orbs + 1) / 2;
    const size_t n2e_elements = n1e_elements * (n1e_elements + 1) / 2;
    
    h1e_.resize(n1e_elements, 0.0);
    h2e_.resize(n2e_elements, 0.0);
}

void IntegralMO::load_from_fcidump(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("IntegralMO: Cannot open FCIDUMP file: " + filename);
    }

    parse_namelist(file);
    
    // Reset stream to parse integral data
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
    size_t h1e_nonzero = 0;
    size_t h2e_nonzero = 0;
    
    for (double val : h1e_) {
        if (std::abs(val) > threshold) {
            ++h1e_nonzero;
        }
    }
    for (double val : h2e_) {
        if (std::abs(val) > threshold) {
            ++h2e_nonzero;
        }
    }
    
    return {h1e_nonzero, h2e_nonzero};
}

void IntegralMO::parse_namelist(std::ifstream& file) {
    std::string line;
    std::string namelist_content;
    bool in_namelist = false;
    
    while (std::getline(file, line)) {
        if (!in_namelist && (line.find("&FCI") != std::string::npos || line.find("&fci") != std::string::npos)) {
            in_namelist = true;
        }
        
        if (in_namelist) {
            namelist_content += line + " ";
            if (line.find("&END") != std::string::npos || line.find("&end") != std::string::npos) {
                break;
            }
        }
    }
    
    if (!in_namelist) {
        throw std::runtime_error("IntegralMO: No &FCI namelist found in FCIDUMP file.");
    }
    
    int file_norb = 0;
    if (extract_parameter(namelist_content, "NORB", file_norb)) {
        if (file_norb != this->n_orbs) {
            throw std::runtime_error("IntegralMO: NORB mismatch: expected " + std::to_string(this->n_orbs) + 
                                     ", but FCIDUMP file has " + std::to_string(file_norb) + " orbitals.");
        }
    }
    
    if (!extract_parameter(namelist_content, "NELEC", n_elecs) || n_elecs < 0) {
        throw std::runtime_error("IntegralMO: Failed to parse valid NELEC from namelist.");
    }
    
    int ms2 = 0;
    if (extract_parameter(namelist_content, "MS2", ms2)) {
        spin_mult = std::abs(ms2) + 1;
    }
}

bool IntegralMO::extract_parameter(const std::string& content, const std::string& param, int& value) {
    std::string upper_content = content;
    std::transform(upper_content.begin(), upper_content.end(), upper_content.begin(), ::toupper);
    
    std::string upper_param = param;
    std::transform(upper_param.begin(), upper_param.end(), upper_param.begin(), ::toupper);

    auto pos = upper_content.find(upper_param + "=");
    if (pos == std::string::npos) {
        return false;
    }
    
    pos += upper_param.length() + 1;
    
    while (pos < content.length() && std::isspace(content[pos])) {
        ++pos;
    }
    
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
    char first_char = line[0];
    for (char c : {'!', '*', 'C', 'c', '#'}) {
        if (first_char == c) {
            return true;
        }
    }
    return false;
}

void IntegralMO::parse_integral_line(std::string& line) {
    // Convert Fortran 'D' exponent to C++ 'E' format
    for (char& c : line) {
        if (c == 'D' || c == 'd') {
            c = 'E';
        }
    }
    
    double value;
    int p, q, r, s;
    std::istringstream iss(line);

    if (!(iss >> value >> p >> q >> r >> s)) {
        return;
    }
    
    // - E_core: value 0 0 0 0
    // - 1e integral: value i j 0 0
    // - 2e integral: value i j k l (Chemists' notation)
    if (p == 0 && q == 0 && r == 0 && s == 0) {
        e_nuc = value;
    } else if (r == 0 && s == 0) {
        // Indices are 1-based in FCIDUMP, convert to 0-based
        h1e_[h1e_key(p - 1, q - 1)] = value;
    } else {
        // Indices are 1-based in FCIDUMP, convert to 0-based
        h2e_[h2e_key(p - 1, q - 1, r - 1, s - 1)] = value;
    }
}

} // namespace lever
