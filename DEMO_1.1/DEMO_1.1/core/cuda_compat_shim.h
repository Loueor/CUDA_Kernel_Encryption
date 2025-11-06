// cuda_compat_shim.h
#pragma once
#include <cuda.h>

/* CUDA 12+ 有些 legacy typedef 不再提供；NVBit 的 generated_cuda_meta.h 仍引用它们。
   这里做最小别名以通过编译。*/

#ifndef CUdeviceptr_v1
typedef CUdeviceptr CUdeviceptr_v1;
#endif

#ifndef CUDA_MEMCPY2D_v1
typedef CUDA_MEMCPY2D CUDA_MEMCPY2D_v1;
#endif

#ifndef CUDA_MEMCPY3D_v1
typedef CUDA_MEMCPY3D CUDA_MEMCPY3D_v1;
#endif

#ifndef CUDA_ARRAY_DESCRIPTOR_v1
typedef CUDA_ARRAY_DESCRIPTOR CUDA_ARRAY_DESCRIPTOR_v1;
#endif

#ifndef CUDA_ARRAY3D_DESCRIPTOR_v1
typedef CUDA_ARRAY3D_DESCRIPTOR CUDA_ARRAY3D_DESCRIPTOR_v1;
#endif
