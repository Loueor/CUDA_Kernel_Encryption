extern "C" __global__ void shellcode() {
    // 内联汇编，强制生成 EXIT 指令
    asm volatile("exit;");
}
int main() { return 0; }