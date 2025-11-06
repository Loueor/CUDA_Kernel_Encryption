#pragma once
#include <cstdint>

// 这些需要 C 符号名（和 NVBit utils 等保持一致）
extern "C" {
    extern int verbose;   // 0/1
    extern int mangled;   // 0/1
}

// 常规 C++ 全局（只声明）
extern uint32_t instr_begin_interval;
extern uint32_t instr_end_interval;

// 在 nvbit_at_init 中调用
void load_tool_env();
