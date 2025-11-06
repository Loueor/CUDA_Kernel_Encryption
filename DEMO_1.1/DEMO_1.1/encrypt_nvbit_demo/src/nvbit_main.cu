#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstdio>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include "../../core/nvbit_tool.h"
#include "../../core/nvbit.h"
#include "../../core/utils/utils.h"
#include "../include/log.hpp"
#include "../include/tool_config.hpp"
#include "../include/elf_utils.hpp"
#include "../include/encrypt.hpp"
#include "../include/func_index.hpp"
#include "../include/helper_decrypt_kernel.hpp"



/* ============ 元数据表 ============ */
static std::mutex meta_mu;
static std::unordered_map<CUmodule, CUcontext> module_owner_ctx;
static std::unordered_map<CUmodule, std::string> module_source;
static std::unordered_map<CUfunction, CUmodule> func_owner_module;

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    GET_VAR_INT(
        instr_begin_interval, "INSTR_BEGIN", 0,
        "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(
        instr_end_interval, "INSTR_END", UINT32_MAX,
        "End of the instruction interval where to apply instrumentation");
    GET_VAR_INT(mangled, "MANGLED_NAMES", 1,
                "Print kernel names mangled or not");

    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");

    init_verbose_once();
    load_tool_env();

    std::string pad(80,'-');
    printf("%s\n", pad.c_str());
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    static std::unordered_map<void*, void*> encrypt_buff_tmp;

     /* 小工具：打印当前 current context */
    auto print_current_ctx = [](const char* where_tag) {
        CUcontext cur = nullptr;
        CUresult r = cuCtxGetCurrent(&cur);
        printf("    [%s] 当前current ctx = %p (cuCtxGetCurrent=%d)\n",
               where_tag, (void*)cur, (int)r);
    };

    if (verbose) {
        std::string event_pad(70, '=');
        printf("\n%s\n", event_pad.c_str());
        printf(" CUDA事件回调触发\n");
        printf(" 上下文: %p\n", (void*)ctx);
        printf(" 状态: %s\n", is_exit ? "退出(EXIT)" : "进入(ENTRY)");
        printf(" 回调ID: %d\n", cbid);
        printf(" API名称: %s\n", name);
        printf(" 参数地址: %p\n", params);

        // 打印线程 ID 信息
        pid_t tid = syscall(SYS_gettid);   // Linux 线程号（系统级）
        pthread_t ptid = pthread_self();   // pthread 线程 ID（用户级）
        printf(" 线程信息: pthread_id=%lu  system_tid=%d\n",
               (unsigned long)ptid, tid);

        //打印当前上下文（统一入口处）
        print_current_ctx(is_exit ? "EXIT" : "ENTRY");

        printf("%s\n", event_pad.c_str());
    }
    if (cbid == API_CUDA_cuLibraryLoadData) {
        if (!is_exit) {
            auto* p = (cuLibraryLoadData_params*)params;
            if (p && p->code) {
                auto [funvec, enc_code] = simple_encrypt_text_region_with_funcs(p->code);
                func_index::append_and_take_ownership(std::move(funvec));
                if (enc_code) {
                    p->code = enc_code;
                    std::lock_guard<std::mutex> lk(meta_mu);
                    encrypt_buff_tmp[params] = enc_code;
                }
            }
        } else {
            std::lock_guard<std::mutex> lk(meta_mu);
            auto it = encrypt_buff_tmp.find(params);
            if (it != encrypt_buff_tmp.end()) {
                free(it->second);
                encrypt_buff_tmp.erase(it);
            }
        }
    }

    if (cbid == API_CUDA_cuLibraryGetKernel && is_exit) {
        auto* p = (cuLibraryGetKernel_params*)params;
        CUfunction f = (p && p->pKernel) ? (CUfunction)(*p->pKernel) : nullptr;
        if (f) {
            CUcontext cur=nullptr; cuCtxGetCurrent(&cur);
            const char* kname = nvbit_get_func_name(cur?cur:ctx, f, mangled);
            uint64_t addr     = nvbit_get_func_addr(cur?cur:ctx, f);
            if (addr && kname) {
                uint64_t len = func_index::get_and_remove_func_len(kname);
                if (len) launch_decrypt_kernel(ctx, addr, (size_t)len);
            }
            printf("[LibraryGetKernel:EXIT] CUfunction=%p name=%s\n",(void*)f,kname?kname:"<null>");
            printf("  address (SASS entry): 0x%016lx\n", addr);
        }
    }

    if (cbid == API_CUDA_cuModuleGetFunction) {
        auto* p = (cuModuleGetFunction_params*)params; // p->hfunc, p->hmod, p->name
        if (!is_exit && verbose) {
            printf("[ModuleGetFunction:ENTRY] mod=%p, name=\"%s\"\n",
                   (void*)p->hmod, p->name ? p->name : "<null>");
            {
                std::lock_guard<std::mutex> lk(meta_mu);
                auto itC = module_owner_ctx.find(p->hmod);
                auto itS = module_source.find(p->hmod);
                if (itC != module_owner_ctx.end()) {
                    printf("模块归属ctx = %p\n", (void*)itC->second);
                }
                if (itS != module_source.end()) {
                    printf("模块来源 = %s\n", itS->second.c_str());
                }
            }
            print_current_ctx("ModuleGetFunction-ENTRY");
        }
        if (is_exit) {
            if (p && p->hfunc && *p->hfunc) {
                {
                    std::lock_guard<std::mutex> lk(meta_mu);
                    func_owner_module[*p->hfunc] = p->hmod;
                }

                CUcontext owner = nullptr;
                std::string src = "<unknown>";
                {
                    std::lock_guard<std::mutex> lk(meta_mu);
                    auto itC = module_owner_ctx.find(p->hmod);
                    if (itC != module_owner_ctx.end()) owner = itC->second;
                    auto itS = module_source.find(p->hmod);
                    if (itS != module_source.end()) src = itS->second;
                }
                 // 3) 为确保能取到地址：必要时临时切到“归属ctx”
                CUcontext cur = nullptr;
                cuCtxGetCurrent(&cur);
                bool need_push = (owner && owner != cur);
                if (need_push) {
                    CUresult r = cuCtxPushCurrent(owner);
                    if (verbose) {
                        printf("[ModuleGetFunction:EXIT] 临时切ctx到归属ctx=%p (push rc=%d)\n",
                            (void*)owner, (int)r);
                        }
                    }
                // 4) 解析名称与 SASS 入口地址（提前打印）
                CUcontext use_ctx = owner ? owner : (cur ? cur : nullptr);
                const char* kname = nvbit_get_func_name(use_ctx, *p->hfunc, mangled);
                uint64_t    addr  = nvbit_get_func_addr(use_ctx, *p->hfunc);

                printf("[ModuleGetFunction:EXIT] CUfunction=%p (requested name=\"%s\")\n",
                    (void*)(*p->hfunc), p->name ? p->name : "<null>");
                printf("来自 CUmodule=%p, 来源=%s\n", (void*)p->hmod, src.c_str());
                printf("模块归属ctx=%p, 当前ctx=%p %s\n",
                   (void*)owner, (void*)cur, (owner && owner != cur) ? "(不一致)" : "");

                // 统一用 nvbit 给到的名字（可与 p->name 校对）
                printf("   ↳ 解析到的函数名: %s\n", kname ? kname : "<null>");
                printf("  address (SASS entry): 0x%016lx%s\n",
                    addr, addr ? "" : "  (addr not ready; will retry at first launch)");
                
                // 5) 恢复上下文
                if (need_push) {
                    CUcontext popped = nullptr;
                    CUresult r = cuCtxPopCurrent(&popped);
                    if (verbose) {
                        printf("[ModuleGetFunction:EXIT] 恢复原ctx=%p (pop rc=%d)\n",
                            (void*)popped, (int)r);
                    }
                }
            } else if (verbose) {
                printf("[ModuleGetFunction:EXIT] 未返回有效函数句柄\n");
            }
        }
    }

    if (cbid == API_CUDA_cuModuleLoad && is_exit) {
        auto* p = (cuModuleLoad_params*)params; // p->module, p->fname
        if (p && p->module && *p->module) {
            CUcontext cur{}; cuCtxGetCurrent(&cur);
            {
                std::lock_guard<std::mutex> lk(meta_mu);
                module_owner_ctx[*p->module] = cur;
                module_source[*p->module] = std::string("file:") + (p->fname ? p->fname : "<null>");
            }
            if (verbose) {
                printf("[ModuleLoad] CUmodule=%p 归属ctx=%p 来源=%s\n",
                       (void*)(*p->module), (void*)cur, module_source[*p->module].c_str());
            }
        }
    }

    if (cbid == API_CUDA_cuModuleLoadData && is_exit) {
        auto* p = (cuModuleLoadData_params*)params; // p->module, p->image
        if (p && p->module && *p->module) {
            CUcontext cur{}; cuCtxGetCurrent(&cur);
            {
                std::lock_guard<std::mutex> lk(meta_mu);
                module_owner_ctx[*p->module] = cur;
                module_source[*p->module] = "data:<image>";
            }
            if (verbose) {
                printf("[ModuleLoadData] CUmodule=%p 归属ctx=%p 来源=%s\n",
                       (void*)(*p->module), (void*)cur, module_source[*p->module].c_str());
            }
        }
    }

    if (cbid == API_CUDA_cuModuleLoadDataEx && is_exit) {
        auto* p = (cuModuleLoadDataEx_params*)params; // p->module, p->image
        if (p && p->module && *p->module) {
            CUcontext cur{}; cuCtxGetCurrent(&cur);
            {
                std::lock_guard<std::mutex> lk(meta_mu);
                module_owner_ctx[*p->module] = cur;
                module_source[*p->module] = "dataex:<image+options>";
            }
            if (verbose) {
                printf("[ModuleLoadDataEx] CUmodule=%p 归属ctx=%p 来源=%s\n",
                       (void*)(*p->module), (void*)cur, module_source[*p->module].c_str());
            }
        }
    }

    if (cbid == API_CUDA_cuModuleLoadFatBinary && is_exit) {
        auto* p = (cuModuleLoadFatBinary_params*)params; // p->module, p->fatCubin
        if (p && p->module && *p->module) {
            CUcontext cur{}; cuCtxGetCurrent(&cur);
            {
                std::lock_guard<std::mutex> lk(meta_mu);
                module_owner_ctx[*p->module] = cur;
                module_source[*p->module] = "fatbin:<image>";
            }
            if (verbose) {
                printf("[ModuleLoadFatBinary] CUmodule=%p 归属ctx=%p 来源=%s\n",
                       (void*)(*p->module), (void*)cur, module_source[*p->module].c_str());
            }
        }
    }

    if (cbid == API_CUDA_cuModuleUnload && !is_exit) {
        auto* p = (cuModuleUnload_params*)params; // p->hmod
        if (verbose) {
            printf("[ModuleUnload:ENTRY] 将卸载 CUmodule=%p\n", (void*)p->hmod);
            std::lock_guard<std::mutex> lk(meta_mu);
            auto itS = module_source.find(p->hmod);
            if (itS != module_source.end()) {
                printf("来源 = %s\n", itS->second.c_str());
            }
        }
        {
            std::lock_guard<std::mutex> lk(meta_mu);
            module_owner_ctx.erase(p->hmod);
            module_source.erase(p->hmod);
            /* 注意：func_owner_module 里残留的函数映射不强清理，查不到时会打印“未知” */
        }
    }

    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel ||
        cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz)

    {
        if (!is_exit)
        {
             sleep(3); 
        }
    }
    else {
        if (verbose) {
            // print_cbid(cbid,name);
        }
    }

    if (verbose) {
        std::string end_pad(70, '=');
        printf("%s\n", end_pad.c_str());
        printf("CUDA事件回调处理完成\n");
        printf("%s\n\n", end_pad.c_str());
    }

}

void nvbit_at_term() {
    printf("NBit run successfully with the address acquisition\n");
    func_index::clear();
}
