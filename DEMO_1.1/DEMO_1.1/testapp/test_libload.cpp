// testapp_driver_load_lib.cpp
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <cuda.h>

#define CK(call) do{ CUresult r=(call); if(r!=CUDA_SUCCESS){ \
  const char* n=nullptr; const char* s=nullptr; \
  cuGetErrorName(r,&n); cuGetErrorString(r,&s); \
  fprintf(stderr,"CUDA error %s: %s at %s:%d\n", n?n:"?", s?s:"<null>", __FILE__, __LINE__); \
  std::exit(1);} }while(0)

static std::vector<unsigned char> read_all(const char* path){
    std::ifstream f(path, std::ios::binary);
    if(!f){ fprintf(stderr,"open %s failed\n", path); std::exit(1); }
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv){
    const char* cubin_path = (argc >= 2) ? argv[1] : "/home/loueor/Desktop/DEMO/testapp/testapp.cubin";

    CK(cuInit(0));
    CUdevice dev; CK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CK(cuCtxSetCurrent(ctx));

    // === 读取 cubin 到内存，并用 cuLibraryLoadData 装载 ===
    auto cubin = read_all(cubin_path);
    printf("[host] cubin size = %zu bytes\n", cubin.size());

    CUlibrary lib = nullptr;

    // 可选：追加 JIT 日志，便于调试（对 cubin 也能给出信息）
    char errlog[8192] = {0}, infolog[4096] = {0};
    CUjit_option o[4]; void* v[4]; unsigned nopt = 0;
    o[nopt] = CU_JIT_ERROR_LOG_BUFFER;            v[nopt++] = errlog;
    o[nopt] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES; v[nopt++] = (void*)(size_t)sizeof(errlog);
    o[nopt] = CU_JIT_INFO_LOG_BUFFER;             v[nopt++] = infolog;
    o[nopt] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;  v[nopt++] = (void*)(size_t)sizeof(infolog);

    CK(cuLibraryLoadData(&lib,
                         (const void*)cubin.data(),
                         o, v, nopt,
                         /*libraryOptions*/nullptr, /*libraryOptionValues*/nullptr, /*numLibraryOptions*/0));

    // === 从库中取内核句柄（用你已有的 mangled 名称，保证匹配）===
    CUkernel kA=nullptr, kB=nullptr, kC=nullptr;
    CK(cuLibraryGetKernel(&kA, lib, "_Z7kernelAPiPKii"));
    CK(cuLibraryGetKernel(&kB, lib, "_Z7kernelBPfPKfS1_fi"));
    CK(cuLibraryGetKernel(&kC, lib, "_Z7kernelCPfPKiPKffi"));

    const int N = 3;

    // ===== A: int -> int =====
    int h_in_i[N]  = {1, 3, 5};
    int h_out_i[N] = {0};
    CUdeviceptr d_in_i = 0, d_out_i = 0;
    CK(cuMemAlloc(&d_in_i,  N * sizeof(int)));
    CK(cuMemAlloc(&d_out_i, N * sizeof(int)));
    CK(cuMemcpyHtoD(d_in_i, h_in_i, N * sizeof(int)));

    void* argsA[] = { &d_out_i, &d_in_i, (void*)&N };
    CK(cuLaunchKernel((CUfunction)kA, 1,1,1, 32,1,1, 0, 0, argsA, nullptr));
    CK(cuMemcpyDtoH(h_out_i, d_out_i, N * sizeof(int)));
    printf("kernelA sample: out_i[0]=%d, out_i[1]=%d, out_i[2]=%d\n",
           h_out_i[0], h_out_i[1], h_out_i[2]);

    // ===== B: float,float -> float =====
    float h_in_f0[N] = {0.0f, 1.0f, 2.0f};
    float h_in_f1[N] = {0.0f, 5.0f,10.0f};
    float h_out_f[N] = {0.0f};
    float alpha = 0.5f;

    CUdeviceptr d_in_f0=0, d_in_f1=0, d_out_f=0;
    CK(cuMemAlloc(&d_in_f0, N * sizeof(float)));
    CK(cuMemAlloc(&d_in_f1, N * sizeof(float)));
    CK(cuMemAlloc(&d_out_f, N * sizeof(float)));
    CK(cuMemcpyHtoD(d_in_f0, h_in_f0, N * sizeof(float)));
    CK(cuMemcpyHtoD(d_in_f1, h_in_f1, N * sizeof(float)));

    void* argsB[] = { &d_out_f, &d_in_f0, &d_in_f1, &alpha, (void*)&N };
    CK(cuLaunchKernel((CUfunction)kB, 1,1,1, 32,1,1, 0, 0, argsB, nullptr));
    CK(cuMemcpyDtoH(h_out_f, d_out_f, N * sizeof(float)));
    printf("kernelB sample: out_f[0]=%.3f, out_f[1]=%.3f, out_f[2]=%.3f\n",
           h_out_f[0], h_out_f[1], h_out_f[2]);

    // ===== C: int + float -> float =====
    int   h_in_i2[N] = {2, 4, 6};
    float h_in_f2[N] = {0.0f, 6.5f, 13.0f};
    float h_out_f2[N] = {0.0f};
    float beta = 1.0f;

    CUdeviceptr d_in_i2=0, d_in_f2=0, d_out_f2=0;
    CK(cuMemAlloc(&d_in_i2, N * sizeof(int)));
    CK(cuMemAlloc(&d_in_f2, N * sizeof(float)));
    CK(cuMemAlloc(&d_out_f2, N * sizeof(float)));
    CK(cuMemcpyHtoD(d_in_i2, h_in_i2, N * sizeof(int)));
    CK(cuMemcpyHtoD(d_in_f2, h_in_f2, N * sizeof(float)));

    void* argsC[] = { &d_out_f2, &d_in_i2, &d_in_f2, &beta, (void*)&N };
    CK(cuLaunchKernel((CUfunction)kC, 1,1,1, 32,1,1, 0, 0, argsC, nullptr));
    CK(cuMemcpyDtoH(h_out_f2, d_out_f2, N * sizeof(float)));
    printf("kernelC sample: out_f2[0]=%.3f, out_f2[1]=%.3f, out_f2[2]=%.3f\n",
           h_out_f2[0], h_out_f2[1], h_out_f2[2]);

    // 资源清理
    cuMemFree(d_in_i);  cuMemFree(d_out_i);
    cuMemFree(d_in_f0); cuMemFree(d_in_f1); cuMemFree(d_out_f);
    cuMemFree(d_in_i2); cuMemFree(d_in_f2); cuMemFree(d_out_f2);

    // 卸载库（注意：用 cuLibraryUnload，而不是 cuModuleUnload）
    CK(cuLibraryUnload(lib));

    // 可选：释放主上下文
    // cuDevicePrimaryCtxRelease(dev);

    if (errlog[0]) fprintf(stderr,"---- JIT ERR ----\n%s\n", errlog);
    if (infolog[0]) fprintf(stderr,"---- JIT INFO ----\n%s\n", infolog);

    return 0;
}
