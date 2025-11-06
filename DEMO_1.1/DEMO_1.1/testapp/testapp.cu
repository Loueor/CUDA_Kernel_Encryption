#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* ========== 设备端函数（包含 extern "C" 与普通 device 函数） ========== */

// 1) extern "C"：整型加法（不被 C++ 名字改编，便于定位）
extern "C" __device__ __noinline__ int add_i(int a, int b) {
    return a + b;
}

// 2) extern "C"：SAXPY（同理不改名）
extern "C" __device__ __noinline__ float saxpy_f(float a, float x, float y) {
    return a * x + y;
}

// 3) 普通 device 函数（可能会被改名）
__device__ __noinline__ int mul_i(int a, int b) {
    return a * b;
}

// 4) inline 混合函数（调用上面两个，制造函数调用链）
__device__ __forceinline__ int mix_int_op(int v) {
    // mul_i 和 add_i 都会被用到，方便在 NVBit 侧抓 related functions
    return mul_i(v, 3) + add_i(v, 2);
}

// 5) 浮点辅助（可能被改名）
__device__ __noinline__ float helper_f(float a, float x, float y) {
    // 故意调用 saxpy_f，形成依赖
    return saxpy_f(a, x, y);
}

/* ========== 多个 Kernel ========== */

// kernelA：对整型数组做 mix 运算
__global__ void kernelA(int* __restrict__ out, const int* __restrict__ in, int n) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) {
        int v = in[id];
        // 使用 mix_int_op -> (mul_i + add_i)
        out[id] = mix_int_op(v);
    }
}

// kernelB：对浮点数组做 SAXPY
__global__ void kernelB(float* __restrict__ out,
                        const float* __restrict__ x,
                        const float* __restrict__ y,
                        float a, int n) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) {
        float xv = x[id];
        float yv = y[id];
        // 走 helper_f -> saxpy_f
        out[id] = helper_f(a, xv, yv);
    }
}

// kernelC：混合整型与浮点（演示第三个 kernel）
__global__ void kernelC(float* __restrict__ out,
                        const int* __restrict__ in_i,
                        const float* __restrict__ in_f,
                        float a, int n) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) {
        // 用 add_i / mul_i 参与计算，再喂给 saxpy_f
        int vi = add_i(in_i[id], mul_i(in_i[id], 2)); // vi = in + 2*in = 3*in
        float r = saxpy_f(a, (float)vi, in_f[id]);    // out = a*vi + in_f
        out[id] = r;
    }
}

/* ========== 实用宏 ========== */
#define CUDA_OK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", \
                #call, __FILE__, __LINE__, cudaGetErrorString(_e)); \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

int main() {
    const int N = 1 << 16; // 65536
    const size_t Ni = N * sizeof(int);
    const size_t Nf = N * sizeof(float);

    int *h_in_i = (int*)malloc(Ni);
    float *h_x = (float*)malloc(Nf);
    float *h_y = (float*)malloc(Nf);
    float *h_out_f = (float*)malloc(Nf);
    int *h_out_i = (int*)malloc(Ni);

    for (int i = 0; i < N; ++i) {
        h_in_i[i] = i % 1000;
        h_x[i] = 0.5f * (float)(i % 1000);
        h_y[i] = 1.0f;
    }

    int *d_in_i = nullptr, *d_out_i = nullptr;
    float *d_x = nullptr, *d_y = nullptr, *d_out_f = nullptr;

    CUDA_OK(cudaMalloc(&d_in_i, Ni));
    CUDA_OK(cudaMalloc(&d_out_i, Ni));
    CUDA_OK(cudaMalloc(&d_x, Nf));
    CUDA_OK(cudaMalloc(&d_y, Nf));
    CUDA_OK(cudaMalloc(&d_out_f, Nf));

    CUDA_OK(cudaMemcpy(d_in_i, h_in_i, Ni, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_x, h_x, Nf, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_y, h_y, Nf, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // 两个 stream，测试并行发射
    cudaStream_t s1, s2;
    CUDA_OK(cudaStreamCreate(&s1));
    CUDA_OK(cudaStreamCreate(&s2));

    // 先在 stream 1 上跑 kernelA（整型）
    kernelA<<<grid, block, 0, s1>>>(d_out_i, d_in_i, N);
    // 再在 stream 2 上跑 kernelB（浮点）
    float a = 2.0f;
    kernelB<<<grid, block, 0, s2>>>(d_out_f, d_x, d_y, a, N);
    
    // 再在 stream 1 上跑 kernelC（第三个 kernel）
    kernelC<<<grid, block, 0, s1>>>(d_out_f, d_in_i, d_x, a, N);

    CUDA_OK(cudaStreamSynchronize(s1));
    CUDA_OK(cudaStreamSynchronize(s2));

    CUDA_OK(cudaMemcpy(h_out_i, d_out_i, Ni, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(h_out_f, d_out_f, Nf, cudaMemcpyDeviceToHost));

    // 简单校验/打印
    printf("kernelA sample: out_i[0]=%d, out_i[1]=%d, out_i[2]=%d\n",
           h_out_i[0], h_out_i[1], h_out_i[2]);
    printf("kernelB/C sample: out_f[0]=%.3f, out_f[1]=%.3f, out_f[2]=%.3f\n",
           h_out_f[0], h_out_f[1], h_out_f[2]);

    CUDA_OK(cudaStreamDestroy(s1));
    CUDA_OK(cudaStreamDestroy(s2));
    CUDA_OK(cudaFree(d_in_i));
    CUDA_OK(cudaFree(d_out_i));
    CUDA_OK(cudaFree(d_x));
    CUDA_OK(cudaFree(d_y));
    CUDA_OK(cudaFree(d_out_f));

    free(h_in_i);
    free(h_x);
    free(h_y);
    free(h_out_f);
    free(h_out_i);

    return 0;
}
