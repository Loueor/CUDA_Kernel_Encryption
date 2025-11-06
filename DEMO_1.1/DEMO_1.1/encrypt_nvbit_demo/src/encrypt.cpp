#include <cstring> 
#include <cinttypes> 
#include "log.hpp"
#include "elf_utils.hpp"
#include "encrypt.hpp"


void* simple_encrypt(const void* raw) {
    if (!raw) return NULL;
    
    // 自动计算cubin长度
    size_t len = get_cubin_code_size(raw);
    if (len == 0) {

        printf("[ENCRYPT] len==0 ");
        return NULL;
    }
    
    printf("[ENCRYPT] Detected cubin size: %zu bytes\n", len);
    

    
    unsigned char* encrypted = (unsigned char*)malloc(len);
    if (!encrypted) return NULL;
    
    const unsigned char* original = (const unsigned char*)raw;
    
    // 复制原始数据
    memcpy(encrypted, original, len);
    printf("[ENCRYPT] memcpy work successfully ");
    
    for (size_t i = 5376; i < 7424; ++i) {
        encrypted[i] = (unsigned char)((original[i] + 1) & 0xFF);
    }
    
    printf("[ENCRYPT] Encrypted bytes over!!!");
    
    return encrypted;
}



enc_result_t simple_encrypt_text_region_StartAndEnd(const void* raw) {
    enc_result_t R = {0};
    if (!raw) return R;

    const unsigned char* data = (const unsigned char*)raw;
    if (!is_valid_elf64_le(data, sizeof(Elf64_Ehdr))) {
        vlog("[enc] not an ELF64 image; pass-through\n");
        return R;
    }
    const Elf64_Ehdr* eh = (const Elf64_Ehdr*)data;

    /* 计算整镜像长度（用于整体拷贝） */
    size_t img_len = get_cubin_code_size(raw);
    if (img_len == 0) {
        vlog("[enc] size detect failed; pass-through\n");
        return R;
    }

    /* 定位节表与节名表 */
    if (!eh->e_shoff || !eh->e_shentsize) {
        vlog("[enc] no section header table; pass-through\n");
        return R;
    }
    if (eh->e_shentsize != sizeof(Elf64_Shdr)) {
        vlog("[enc] shentsize mismatch; pass-through\n");
        return R;
    }

    const Elf64_Shdr* sh_base = (const Elf64_Shdr*)(data + eh->e_shoff);
    uint16_t shnum = eh->e_shnum;
    if (shnum == 0) {
        /* 扩展节计数：sh_base[0].sh_size 存放实际节数 */
        shnum = (uint16_t)sh_base[0].sh_size;
        if (!shnum) {
            vlog("[enc] extended section count is 0; pass-through\n");
            return R;
        }
    }

    if (eh->e_shstrndx == 0 || eh->e_shstrndx >= shnum) {
        vlog("[enc] invalid e_shstrndx; pass-through\n");
        return R;
    }
    const Elf64_Shdr* shstr = &sh_base[eh->e_shstrndx];

    /* 找到所有 .text* 可执行段的最小/最大文件偏移 */
    uint64_t min_off = UINT64_MAX;
    uint64_t max_end = 0;
    for (uint16_t i = 0; i < shnum; ++i) {
        const Elf64_Shdr* s = &sh_base[i];
        const char* sname = get_sh_name(data, eh, shstr, s);
        if (is_exec_text_section(sname, s)) {
            uint64_t off = s->sh_offset;
            uint64_t end = s->sh_offset + s->sh_size;
            if (off < min_off) min_off = off;
            if (end > max_end) max_end = end;
            if (g_verbose) {
                fprintf(stderr, "[enc] text-sec: idx=%u name=%s off=%" PRIu64 " size=%" PRIu64 "\n",
                        i, sname ? sname:"<noname>", s->sh_offset, s->sh_size);
            }
        }
    }

    if (min_off == UINT64_MAX || max_end <= min_off) {
        vlog("[enc] no .text* executable section found; pass-through\n");
        return R;
    }

    /* 分配拷贝 + 只在 [start,end) 做 +1 处理 */
    unsigned char* out = (unsigned char*)malloc(img_len);
    if (!out) {
        vlog("[enc] malloc failed\n");
        return R;
    }
    memcpy(out, data, img_len);

    size_t start = (size_t)min_off;
    size_t end   = (size_t)max_end;
    if (end > img_len) end = img_len; /* 防御性处理 */

    for (size_t i = start; i < end; ++i) {
        out[i] = (unsigned char)((out[i] + 0) & 0xFF);//TODO
    }

    if (g_verbose) {
        fprintf(stderr, "[enc] encrypt range: start=%zu end=%zu len=%zu (image=%zu)\n",
                start, end, end - start, img_len);
    }

    R.ptr   = out;
    R.start = start;
    R.end   = end;
    R.len   = img_len;
    return R;
}

std::pair<std::vector<FuncLen>, unsigned char*>
simple_encrypt_text_region_with_funcs(const void* raw) {
    std::vector<FuncLen> funs;
    unsigned char* out = nullptr;

    if (!raw) return {funs, out};

    const unsigned char* data = (const unsigned char*)raw;
    if (!is_valid_elf64_le(data, sizeof(Elf64_Ehdr))) {
        vlog("[enc] not an ELF64 image; pass-through\n");
        return {funs, out};
    }
    const Elf64_Ehdr* eh = (const Elf64_Ehdr*)data;

    // 1) 计算整镜像长度（用于整体拷贝）
    size_t img_len = get_cubin_code_size(raw);
    if (img_len == 0) {
        vlog("[enc] size detect failed; pass-through\n");
        return {funs, out};
    }

    // 2) 先收集 (funname, .text_length)
    funs = collect_mainfunc_text_lengths(data, img_len);

    // 打印函数向量（你要求“print the struct”）
    fprintf(stderr, "\n=== (funname, .text_length) ===\n");
    for (const auto& f : funs) {
        fprintf(stderr, "mangled=%-60s  len=%" PRIu64 "\n",
                f.name_mangled.c_str(),
                (unsigned long long)f.text_size);
    }
    fprintf(stderr, "=== total %zu functions ===\n\n", funs.size());

    // 3) 定位节名表、扫所有 .text* 可执行节，找加密范围 [min_off, max_end)
    if (!eh->e_shoff || !eh->e_shentsize) {
        vlog("[enc] no section header table; pass-through\n");
        return {funs, out};
    }
    if (eh->e_shentsize != sizeof(Elf64_Shdr)) {
        vlog("[enc] shentsize mismatch; pass-through\n");
        return {funs, out};
    }

    const Elf64_Shdr* sh_base = (const Elf64_Shdr*)(data + eh->e_shoff);
    uint16_t shnum = eh->e_shnum;
    if (shnum == 0) { // 扩展节计数
        shnum = (uint16_t)sh_base[0].sh_size;
        if (!shnum) {
            vlog("[enc] extended section count is 0; pass-through\n");
            return {funs, out};
        }
    }

    if (eh->e_shstrndx == 0 || eh->e_shstrndx >= shnum) {
        vlog("[enc] invalid e_shstrndx; pass-through\n");
        return {funs, out};
    }
    const Elf64_Shdr* shstr = &sh_base[eh->e_shstrndx];

    uint64_t min_off = UINT64_MAX;
    uint64_t max_end = 0;
    for (uint16_t i = 0; i < shnum; ++i) {
        const Elf64_Shdr* s = &sh_base[i];
        const char* sname = get_sh_name(data, eh, shstr, s);
        if (is_exec_text_section(sname, s)) {
            uint64_t off = s->sh_offset;
            uint64_t end = s->sh_offset + s->sh_size;
            if (off < min_off) min_off = off;
            if (end > max_end) max_end = end;
            if (g_verbose) {
                fprintf(stderr, "[enc] text-sec: idx=%u name=%s off=%" PRIu64 " size=%" PRIu64 "\n",
                        i, sname ? sname : "<noname>", s->sh_offset, s->sh_size);
            }
        }
    }

    if (min_off == UINT64_MAX || max_end <= min_off) {
        vlog("[enc] no .text* executable section found; pass-through\n");
        return {funs, out};
    }

    // 4) 分配拷贝 + 对 [start,end) 进行“加密”
    out = (unsigned char*)malloc(img_len);
    if (!out) {
        vlog("[enc] malloc failed\n");
        return {funs, nullptr};
    }
    memcpy(out, data, img_len);

    size_t start = (size_t)min_off;
    size_t end   = (size_t)max_end;
    if (end > img_len) end = img_len; // 防御性

    // 简单“加密”：字节 +1（你可以改为 XOR、轮换等）
    for (size_t i = start; i < end; ++i) {
        out[i] = (unsigned char)((out[i] + 1) & 0xFF);
    }

    if (g_verbose) {
        fprintf(stderr, "[enc] encrypt range: start=%zu end=%zu len=%zu (image=%zu)\n",
                start, end, end - start, img_len);
    }

    return {funs, out};
}
