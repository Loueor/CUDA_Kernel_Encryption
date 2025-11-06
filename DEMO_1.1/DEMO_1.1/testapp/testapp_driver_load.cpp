// testapp_driver_load.cpp
// testapp_driver_load.cpp
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cuda.h>

#define CK(call) do{ CUresult r=(call); if(r!=CUDA_SUCCESS){ \
  const char* s=nullptr; cuGetErrorString(r,&s); \
  fprintf(stderr,"CUDA error %d: %s at %s:%d\n",r,s?s:"<null>",__FILE__,__LINE__); exit(1);} }while(0)

int main(){
    CK(cuInit(0));
    CUdevice dev; CK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CK(cuCtxSetCurrent(ctx));

    CUmodule mod; 
    CK(cuModuleLoad(&mod, "/home/loueor/Desktop/DEMO/testapp/testapp.cubin"));

    CUfunction kA, kB, kC;
    CK(cuModuleGetFunction(&kA, mod, "_Z7kernelAPiPKii"));
    CK(cuModuleGetFunction(&kB, mod, "_Z7kernelBPfPKfS1_fi"));
    CK(cuModuleGetFunction(&kC, mod, "_Z7kernelCPfPKiPKffi"));

    const int N = 3;

    // ===== A: int -> int =====
    int h_in_i[N]  = {1, 3, 5};
    int h_out_i[N] = {0};
    CUdeviceptr d_in_i = 0, d_out_i = 0;
    CK(cuMemAlloc(&d_in_i,  N * sizeof(int)));
    CK(cuMemAlloc(&d_out_i, N * sizeof(int)));
    CK(cuMemcpyHtoD(d_in_i, h_in_i, N * sizeof(int)));

    void* argsA[] = { &d_out_i, &d_in_i, (void*)&N };
    CK(cuLaunchKernel(kA, 1,1,1, 32,1,1, 0, 0, argsA, nullptr));
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
    CK(cuLaunchKernel(kB, 1,1,1, 32,1,1, 0, 0, argsB, nullptr));
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
    CK(cuLaunchKernel(kC, 1,1,1, 32,1,1, 0, 0, argsC, nullptr));
    CK(cuMemcpyDtoH(h_out_f2, d_out_f2, N * sizeof(float)));
    printf("kernelC sample: out_f2[0]=%.3f, out_f2[1]=%.3f, out_f2[2]=%.3f\n",
           h_out_f2[0], h_out_f2[1], h_out_f2[2]);

    // 清理（可选）
    cuMemFree(d_in_i);  cuMemFree(d_out_i);
    cuMemFree(d_in_f0); cuMemFree(d_in_f1); cuMemFree(d_out_f);
    cuMemFree(d_in_i2); cuMemFree(d_in_f2); cuMemFree(d_out_f2);
    // cuModuleUnload(mod); cuDevicePrimaryCtxRelease(dev);

    return 0;
}

