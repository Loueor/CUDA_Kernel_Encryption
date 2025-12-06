import ctypes
import numpy as np
import os

# =========================================================
# 终极方案：手动加载 CUDA 运行时库 (无需 pip install cuda-python)
# =========================================================

# 1. 尝试加载 libcudart.so (CUDA Runtime)
# Linux 下通常在 /usr/local/cuda/lib64 或 LD_LIBRARY_PATH 中
try:
    # 常见路径尝试
    candidates = [
        'libcudart.so', 
        'libcudart.so.11.0', 
        'libcudart.so.12',
        '/usr/local/cuda/lib64/libcudart.so'
    ]
    cudart = None
    for lib in candidates:
        try:
            cudart = ctypes.CDLL(lib)
            print(f"[Server] Loaded CUDA Runtime: {lib}")
            break
        except OSError:
            continue
            
    if cudart is None:
        raise OSError("Could not load libcudart.so. Please ensure CUDA Toolkit is installed and in LD_LIBRARY_PATH.")

except Exception as e:
    print(f"[Error] Failed to load CUDA runtime: {e}")
    exit(1)

# 定义 cudart 函数原型
# cudaMalloc(void **devPtr, size_t size)
cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cudart.cudaMalloc.restype = ctypes.c_int

# cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
cudart.cudaMemcpy.restype = ctypes.c_int

# cudaMemset(void *devPtr, int value, size_t count)
cudart.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
cudart.cudaMemset.restype = ctypes.c_int

# cudaDeviceSynchronize()
cudart.cudaDeviceSynchronize.argtypes = []
cudart.cudaDeviceSynchronize.restype = ctypes.c_int

# cudaMemcpyKind 枚举值
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2

# =========================================================
# 加载漏洞算子库
# =========================================================
if not os.path.exists('./libllm_kernel.so'):
    print("[Error] libllm_kernel.so not found! Did you compile it?")
    exit(1)

dll = ctypes.CDLL('./libllm_kernel.so')
# void embedding_kernel(int* input, int* output, int len)
dll.embedding_kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

class SafeAICloud:
    def __init__(self):
        print("[Server] Initializing LLM Engine (No-Lib ver)...")
        self.d_input = ctypes.c_void_p(0)
        self.d_output = ctypes.c_void_p(0)
        
        # 分配 4KB 显存
        check(cudart.cudaMalloc(ctypes.byref(self.d_input), 4096))
        check(cudart.cudaMalloc(ctypes.byref(self.d_output), 4096))
        
        # 初始化 Output
        check(cudart.cudaMemset(self.d_output, 0, 4096))

    def inference(self, user_input_list):
        input_len = len(user_input_list)
        input_array = np.array(user_input_list, dtype=np.int32)
        
        # 1. 拷贝数据 Host -> Device
        check(cudart.cudaMemcpy(self.d_input, input_array.ctypes.data, input_array.nbytes, cudaMemcpyHostToDevice))
        
        # 2. 调用漏洞 Kernel
        dll.embedding_kernel(self.d_input, self.d_output, input_len)
        
        # 3. 同步
        check(cudart.cudaDeviceSynchronize())
        
        # 4. 拷贝结果 Device -> Host
        h_output = np.zeros(16, dtype=np.int32)
        check(cudart.cudaMemcpy(h_output.ctypes.data, self.d_output, h_output.nbytes, cudaMemcpyDeviceToHost))
        
        # 获取指针的整数值 (用于 Payload 构造)
        return h_output, self.d_input.value, self.d_output.value

def check(err):
    if err != 0:
        raise Exception(f"CUDA Error Code: {err}")

# 全局服务实例
service = SafeAICloud()