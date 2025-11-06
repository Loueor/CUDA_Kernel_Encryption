#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <stdio.h>



extern "C" __global__ void decrypt_bytes_kernel(unsigned char* base, int n) {
    // 移除const，允许修改
    for (int i = 0; i < n; i++) {
        base[i] = base[i] - 1;  // 直接修改原数据
        printf("%02x ", base[i] & 0xff);
        
        if ((i + 1) % 16 == 0) {
            printf("\n");
        }
    }
    printf("print num is = %d\n", n);
}
