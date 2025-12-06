#include <stdio.h>

// 模拟显存中的“绝密权重”
// 在真实场景中，这通常通过指针传递，这里为了简化演示，我们假设攻击者想把输出全改成这个值
__device__ unsigned int SECRET_WEIGHT = 0xCAFEBABE; 

// -----------------------------------------------------------
// 1. 包含漏洞的自定义算子 (Vulnerable Operator)
// -----------------------------------------------------------
// 这是一个模拟 "Embedding Preprocessing" 的函数
// 开发者为了性能，使用了递归和栈内存，且没有做边界检查
__device__ void preprocess_embedding(int* input_tokens, int* output_buffer, int len, int depth) {
    // [漏洞点]：栈上的固定大小缓冲区
    int local_cache[16]; 
    
    // 初始化
    local_cache[0] = 0x11111111;

    // [漏洞点]：没有检查 len 是否超过 16
    // 攻击者可以通过控制 len 覆盖栈上的返回地址
    for (int i = 0; i < len; ++i) {
        local_cache[i] = input_tokens[i];
    }

    // 递归调用，强制编译器将返回地址 (R20) 压入栈中 (Spill to Stack)
    // 这是攻击成功的必要条件
    if (depth > 0) {
        preprocess_embedding(input_tokens, output_buffer, len, depth - 1);
    }
}

// -----------------------------------------------------------
// 2. Kernel 入口
// -----------------------------------------------------------
extern "C" __global__ void embedding_kernel(int* input_tokens, int* output_buffer, int len) {
    // 正常业务逻辑：预处理 Token
    preprocess_embedding(input_tokens, output_buffer, len, 1);
}