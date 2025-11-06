#pragma once
#include "../../core/cuda.h"
#include <cstddef>
#include <cstdint>

// 外部符号（由 objcopy 嵌入 fatbin 产生）
extern "C" {
    extern unsigned char _binary_decrypt_bytes_kernel_fatbin_start[];
    extern unsigned char _binary_decrypt_bytes_kernel_fatbin_end[];
    extern unsigned long _binary_decrypt_bytes_kernel_fatbin_size;
}

struct CtxRes {
    CUmodule   mod   = nullptr;
    CUfunction fun   = nullptr;
    CUstream   stream= nullptr;
    bool       inited= false;
};

bool ensure_ctx_inited(CUcontext ctx_unused = nullptr);
void launch_decrypt_kernel(CUcontext ctx, size_t addr, size_t n);
