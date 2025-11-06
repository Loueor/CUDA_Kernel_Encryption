#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>
#include "elf_utils.hpp"

struct enc_result_t {
    void*  ptr;
    size_t start;
    size_t end;
    size_t len;
};

void* simple_encrypt(const void* raw);

enc_result_t simple_encrypt_text_region_StartAndEnd(const void* raw);

// 返回 (函数向量, 加密后的新buffer)。调用者负责 free 后者。
std::pair<std::vector<FuncLen>, unsigned char*>
simple_encrypt_text_region_with_funcs(const void* raw);
