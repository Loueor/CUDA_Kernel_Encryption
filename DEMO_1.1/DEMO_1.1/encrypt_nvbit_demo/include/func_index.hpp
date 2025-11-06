#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "elf_utils.hpp"

// 拥有 funcs_lens 的唯一权；避免到处 extern 全局变量
namespace func_index {
    void append_and_take_ownership(std::vector<FuncLen>&& fv);
    uint64_t get_and_remove_func_len(const std::string& kname);
    void clear();
}
