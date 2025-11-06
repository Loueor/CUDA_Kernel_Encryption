#include "helper_decrypt_kernel.hpp"
#include "log.hpp"
#include <unordered_map>
#include <mutex>

static std::mutex g_mu;
static std::unordered_map<CUcontext, CtxRes> g_ctxres;
static std::mutex g_mu_init;
static std::unordered_map<CUcontext, bool> g_inited;

bool ensure_ctx_inited(CUcontext /*ctx_unused*/) {
    CUcontext ctx=nullptr;
    if (cuCtxGetCurrent(&ctx)!=CUDA_SUCCESS || !ctx) return false;

    { std::lock_guard<std::mutex> lk(g_mu_init);
      if (g_inited.count(ctx)) return true; }

    CtxRes cr{};
    const void* fatbin_ptr = (const void*)_binary_decrypt_bytes_kernel_fatbin_start;
    if (cuModuleLoadData(&cr.mod, fatbin_ptr) != CUDA_SUCCESS) return false;
    if (cuModuleGetFunction(&cr.fun, cr.mod, "decrypt_bytes_kernel") != CUDA_SUCCESS) return false;
    if (cuStreamCreate(&cr.stream, CU_STREAM_DEFAULT) != CUDA_SUCCESS) return false;
    cr.inited = true;

    { std::lock_guard<std::mutex> lk(g_mu); g_ctxres[ctx]=cr; }
    { std::lock_guard<std::mutex> lk(g_mu_init); g_inited[ctx]=true; }
    return true;
}

void launch_decrypt_kernel(CUcontext ctx, size_t addr, size_t n) {
    if (!ctx || !n) return;
    if (!ensure_ctx_inited(ctx)) { fprintf(stderr,"[helper] ctx=%p not initialized\n",(void*)ctx); return; }

    CtxRes cr{};
    { std::lock_guard<std::mutex> lk(g_mu);
      auto it=g_ctxres.find(ctx);
      if (it==g_ctxres.end() || !it->second.inited) { fprintf(stderr,"[helper] ctx=%p no CtxRes\n",(void*)ctx); return; }
      cr = it->second; }

    unsigned long long base_param = (unsigned long long)addr;
    size_t n_param = n;
    void* args[2] = { &base_param, &n_param };

    CUresult rc = cuLaunchKernel(cr.fun, 1,1,1, 1,1,1, 0, cr.stream, args, nullptr);
    if (rc != CUDA_SUCCESS) return;
    cuStreamSynchronize(cr.stream);
}
