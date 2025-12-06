import re
import subprocess
import sys

# 运行 cuobjdump
result = subprocess.run(['cuobjdump', '-sass', 'shellcode_gen.cubin'], capture_output=True, text=True)
output = result.stdout

print("Extracting Shellcode bytes...")

# 简单的正则提取 (提取 /* 0x... */ 中的 16进制)
insts = re.findall(r'/\* (0x[0-9a-f]+) \*/', output)

c_array = []
for inst in insts:
    # 转换为 long long 格式
    c_array.append(int(inst, 16))

print(f"Found {len(insts)//2} instructions (128-bit each).")
print("Use this array in your attack script:")
print("-" * 20)
print("shellcode_raw = [")
for i in range(0, len(c_array), 2):
    # 两个 64位 组成一个 128位 指令，注意小端序组合
    # SASS 显示的是 高位在前，但内存布局通常是低位在前，这里我们直接存原始数值
    # 实际上我们直接按 int64 数组写入即可，顺序按 cuobjdump 的顺序
    print(f"    {hex(c_array[i+1])}, {hex(c_array[i])}, # Inst {i//2}")
print("]")
print("-" * 20)