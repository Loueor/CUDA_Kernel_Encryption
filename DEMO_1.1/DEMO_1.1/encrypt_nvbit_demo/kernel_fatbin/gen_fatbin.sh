#!/usr/bin/env bash
set -euo pipefail
# 生成 fatbin 并用 objcopy 嵌入为可链接目标
# 需要 nvcc 与 objcopy 可用

ROOT=$(dirname "$0")/..
KERN=$ROOT/kernels/decrypt_kernel.cu
OUT=$ROOT/fatbin/decrypt_bytes_kernel.fatbin
OBJ=$ROOT/fatbin/decrypt_bytes_kernel.fatbin.o

# 生成 fatbin
nvcc -fatbin -arch=sm_86 -o "$OUT" "$KERN"

# 嵌入为对象文件，生成符号 _binary_read_bytes_kernel_fatbin_{start,end,size}
objcopy -I binary -O elf64-x86-64 -B i386:x86-64 \
  --redefine-sym _binary_decrypt_bytes_kernel_fatbin_start=_binary_decrypt_bytes_kernel_fatbin_start \
  --redefine-sym _binary_decrypt_bytes_kernel_fatbin_end=_binary_decrypt_bytes_kernel_fatbin_end \
  --redefine-sym _binary_decrypt_bytes_kernel_fatbin_size=_binary_decrypt_bytes_kernel_fatbin_size \
  "$OUT" "$OBJ"

echo "[fatbin] generated: $OBJ"
