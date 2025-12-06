#include <stdio.h>

extern "C" __global__ void shellcode(int* target_addr) {
    // 攻击意图：将目标地址的值改为 0xDEADBEEF (代表被黑了)
    // 在真实攻击中，这里可以是读取 target_addr 的内容并回传
    *target_addr = 0xDEADBEEF; 
    
    // 执行完坏事后，干净地退出，防止 GPU 崩溃报警
    asm volatile("exit;");
}

int main() { return 0; }