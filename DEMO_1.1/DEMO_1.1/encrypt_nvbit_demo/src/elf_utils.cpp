#include <cstring> 
#include <cinttypes> 
#include "elf_utils.hpp"
#include "log.hpp"

const char* sh_type_name(uint32_t t) {
    switch (t) {
        case 0:  return "NULL";
        case 1:  return "PROGBITS";
        case 2:  return "SYMTAB";
        case 3:  return "STRTAB";
        case 4:  return "RELA";
        case 5:  return "HASH";
        case 6:  return "DYNAMIC";
        case 7:  return "NOTE";
        case 8:  return "NOBITS";
        case 9:  return "REL";
        case 10: return "SHLIB";
        case 11: return "DYNSYM";
        default:
            if (t >= 0x70000000 && t <= 0x7FFFFFFF) return "PROC/OS (vendor)";
            return "Unknown";
    }
}

/* print section header flags*/
void print_sh_flags(uint64_t flg) {
    if (!g_verbose) return;
    fprintf(stderr, "    Flags: 0x%016" PRIx64 " [", flg);
    int first = 1;
    if (flg & 0x1)  { fprintf(stderr, "%sWRITE",      first?"":"|"); first = 0; }   /* SHF_WRITE   */
    if (flg & 0x2)  { fprintf(stderr, "%sALLOC",      first?"":"|"); first = 0; }   /* SHF_ALLOC   */
    if (flg & 0x4)  { fprintf(stderr, "%sEXECINSTR",  first?"":"|"); first = 0; }   /* SHF_EXECINSTR */
    if (flg & 0x10) { fprintf(stderr, "%sMERGE",      first?"":"|"); first = 0; }
    if (flg & 0x20) { fprintf(stderr, "%sSTRINGS",    first?"":"|"); first = 0; }
    if (flg & 0x40) { fprintf(stderr, "%sINFO_LINK",  first?"":"|"); first = 0; }
    if (flg & 0x80) { fprintf(stderr, "%sLINK_ORDER", first?"":"|"); first = 0; }
    if (flg & 0x100){ fprintf(stderr, "%sOS_NONCONFORMING", first?"":"|"); first = 0; }
    if (flg & 0x200){ fprintf(stderr, "%sGROUP",      first?"":"|"); first = 0; }
    if (flg & 0x400){ fprintf(stderr, "%sTLS",        first?"":"|"); first = 0; }
    fprintf(stderr, "]\n");
}

/* print program header type name*/
const char* ph_type_name(uint32_t t) {
    switch (t) {
        case 0:  return "PT_NULL";
        case 1:  return "PT_LOAD";
        case 2:  return "PT_DYNAMIC";
        case 3:  return "PT_INTERP";
        case 4:  return "PT_NOTE";
        case 5:  return "PT_SHLIB";
        case 6:  return "PT_PHDR";
        case 7:  return "PT_TLS";
        case 0x6474e550: return "PT_GNU_EH_FRAME";
        case 0x6474e551: return "PT_GNU_STACK";
        default:
            if (t >= 0x70000000 && t <= 0x7FFFFFFF) return "PT_PROC/OS (vendor)";
            return "PT_Unknown";
    }
}

/* print program header flags*/
void print_ph_flags(uint32_t flg) {
    if (!g_verbose) return;
    /* PF_X=1, PF_W=2, PF_R=4 */
    fprintf(stderr, "    Flags: 0x%08" PRIx32 " [", flg);
    int first = 1;
    if (flg & 0x4) { fprintf(stderr, "%sREAD",    first?"":"|"); first = 0; }
    if (flg & 0x2) { fprintf(stderr, "%sWRITE",   first?"":"|"); first = 0; }
    if (flg & 0x1) { fprintf(stderr, "%sEXECUTE", first?"":"|"); first = 0; }
    fprintf(stderr, "]\n");
}

/*to justify the elf64 valid or not*/
bool is_valid_elf64_le(const unsigned char* data, size_t min_len) {
    init_verbose_once();

    if (g_verbose) {
        fprintf(stderr, "\n=== ELF Format Validation ===\n");
        fprintf(stderr, "[ELF_CHECK] Starting ELF validation...\n");
    }

    if (!data) {
        if (g_verbose) fprintf(stderr, "[ELF_CHECK] ERROR: data pointer is NULL\n\n");
        return 0;
    }
    if (min_len < sizeof(Elf64_Ehdr)) {
        if (g_verbose) {
            fprintf(stderr, "[ELF_CHECK] ERROR: min_len (%zu) < ELF header size (%zu)\n\n",
                    min_len, sizeof(Elf64_Ehdr));
        }
        return 0;
    }
    if (!(data[0]==0x7F && data[1]=='E' && data[2]=='L' && data[3]=='F')) {
        if (g_verbose) {
            fprintf(stderr, "[ELF_CHECK] ERROR: Invalid ELF magic: %02X %02X %02X %02X (expected 7F 45 4C 46)\n\n",
                    data[0], (unsigned char)data[1], (unsigned char)data[2], (unsigned char)data[3]);
        }
        return 0;
    }

    const Elf64_Ehdr* eh = (const Elf64_Ehdr*)data;

    if (eh->e_ident[4] != 2 /*ELFCLASS64*/) {
        if (g_verbose) fprintf(stderr, "[ELF_CHECK] ERROR: Not 64-bit ELF (EI_CLASS=%u, expect 2)\n\n",
                               eh->e_ident[4]);
        return 0;
    }
    if (eh->e_ident[5] != 1 /*ELFDATA2LSB*/) {
        if (g_verbose) fprintf(stderr, "[ELF_CHECK] ERROR: Not little-endian (EI_DATA=%u, expect 1)\n\n",
                               eh->e_ident[5]);
        return 0;
    }
    if (eh->e_ehsize != sizeof(Elf64_Ehdr)) {
        if (g_verbose) fprintf(stderr, "[ELF_CHECK] ERROR: e_ehsize=%u mismatch (expect %zu)\n\n",
                               eh->e_ehsize, sizeof(Elf64_Ehdr));
        return 0;
    }

    if (g_verbose) {
        /* 机器类型补充 EM_CUDA */
        const char* mach = "Unknown";
        switch (eh->e_machine) {
            case 0x3:   mach = "x86"; break;
            case 0x3E:  mach = "x86-64"; break;
            case 0x28:  mach = "ARM"; break;
            case 0xB7:  mach = "AArch64"; break;
            case 0xBE:  mach = "CUDA (EM_CUDA)"; break; /* 重要：CUDA cubin 常见 */
        }

        fprintf(stderr, "[ELF_CHECK] SUCCESS: Valid 64-bit little-endian ELF file\n");
        fprintf(stderr, "[ELF_CHECK] ELF Header Details:\n");
        fprintf(stderr, "  Type: %u", eh->e_type);
        switch (eh->e_type) {
            case 1: fprintf(stderr, " (Relocatable)\n"); break;
            case 2: fprintf(stderr, " (Executable)\n"); break;
            case 3: fprintf(stderr, " (Shared Library)\n"); break;
            case 4: fprintf(stderr, " (Core Dump)\n"); break;
            default: fprintf(stderr, " (Unknown)\n"); break;
        }
        fprintf(stderr, "  Machine: 0x%X (%s)\n", eh->e_machine, mach);
        fprintf(stderr, "  Version: %u\n", eh->e_version);
        fprintf(stderr, "  Entry Point: 0x%016" PRIx64 "\n", eh->e_entry);
        fprintf(stderr, "  Program Headers: offset=%" PRIu64 ", entry_size=%u, count=%u\n",
                eh->e_phoff, eh->e_phentsize, eh->e_phnum);
        fprintf(stderr, "  Section Headers: offset=%" PRIu64 ", entry_size=%u, count=%u\n",
                eh->e_shoff, eh->e_shentsize, eh->e_shnum);
        fprintf(stderr, "  Section Name String Table Index: %u\n", eh->e_shstrndx);
        fprintf(stderr, "=== ELF Validation Complete ===\n\n");
    }
    return 1;
}

/* get the max section offset */
size_t size_from_sections(const unsigned char* data, const Elf64_Ehdr* eh) {
    init_verbose_once();
    if (g_verbose) fprintf(stderr, "\n=== Calculating Size from Sections ===\n");

    if (!eh->e_shoff) {
        if (g_verbose) fprintf(stderr, "[SIZE_SECTIONS] ERROR: No section header table (e_shoff=0)\n\n");
        return 0;
    }
    if (!eh->e_shentsize) {
        if (g_verbose) fprintf(stderr, "[SIZE_SECTIONS] ERROR: Invalid e_shentsize=0\n\n");
        return 0;
    }
    if (eh->e_shentsize != sizeof(Elf64_Shdr)) {
        if (g_verbose) {
            fprintf(stderr, "[SIZE_SECTIONS] ERROR: e_shentsize=%u mismatch (expect %zu)\n\n",
                    eh->e_shentsize, sizeof(Elf64_Shdr));
        }
        return 0;
    }

    const Elf64_Shdr* sh0 = (const Elf64_Shdr*)(data + eh->e_shoff);
    uint16_t shnum = eh->e_shnum;
    if (shnum == 0) { /* 扩展节计数 */
        shnum = (uint16_t)sh0->sh_size;
        if (g_verbose) fprintf(stderr, "[SIZE_SECTIONS] Using extended section count: %u\n", shnum);
        if (!shnum) {
            if (g_verbose) fprintf(stderr, "[SIZE_SECTIONS] ERROR: Extended section count is 0\n\n");
            return 0;
        }
    } else {
        if (g_verbose) fprintf(stderr, "[SIZE_SECTIONS] Section count from header: %u\n", shnum);
    }

    if (g_verbose) {
        fprintf(stderr, "[SIZE_SECTIONS] Section header table at offset: %" PRIu64 "\n", eh->e_shoff);
        fprintf(stderr, "[SIZE_SECTIONS] Scanning %u sections to find file size...\n", shnum);
    }

    size_t max_end = 0;
    int processed = 0, skipped_nobits = 0;

    for (uint16_t i = 0; i < shnum; ++i) {
        const Elf64_Shdr* s = (const Elf64_Shdr*)(data + eh->e_shoff + (size_t)i * eh->e_shentsize);

        if (s->sh_type == 8 /*SHT_NOBITS*/) {
            if (g_verbose)
                fprintf(stderr, "[SIZE_SECTIONS] Section[%u]: SKIP NOBITS (no file data)\n", i);
            skipped_nobits++;
            continue;
        }

        uint64_t off = s->sh_offset;
        uint64_t sz  = s->sh_size;
        if (sz > SIZE_MAX - off) {
            if (g_verbose) {
                fprintf(stderr, "[SIZE_SECTIONS] ERROR: Section[%u] overflow: off=%" PRIu64 ", size=%" PRIu64 "\n\n",
                        i, off, sz);
            }
            return 0;
        }

        size_t end = (size_t)(off + sz);
        if (end > max_end) {
            if (g_verbose) {
                fprintf(stderr, "[SIZE_SECTIONS] Section[%u]: NEW MAX END = %zu\n", i, end);
                fprintf(stderr, "    Offset: %" PRIu64 ", Size: %" PRIu64 ", Type: 0x%X (%s)\n",
                        off, sz, s->sh_type, sh_type_name(s->sh_type));
                print_sh_flags(s->sh_flags);
            }
            max_end = end;
        } else if (g_verbose) {
            fprintf(stderr, "[SIZE_SECTIONS] Section[%u]: End = %zu (offset=%" PRIu64 ", size=%" PRIu64 ", type=0x%X/%s)\n",
                    i, end, off, sz, s->sh_type, sh_type_name(s->sh_type));
        }
        processed++;
    }

    if (g_verbose) {
        fprintf(stderr, "[SIZE_SECTIONS] SUMMARY: Processed %d sections, skipped %d NOBITS\n",
                processed, skipped_nobits);
        fprintf(stderr, "[SIZE_SECTIONS] FINAL SIZE from sections: %zu bytes\n", max_end);
        fprintf(stderr, "=== Section Size Calculation Complete ===\n\n");
    }
    return max_end;
}

/* get the max segements offset */
size_t size_from_segments(const unsigned char* data, const Elf64_Ehdr* eh) {
    init_verbose_once();
    if (g_verbose) fprintf(stderr, "\n=== Calculating Size from Segments ===\n");

    if (!eh->e_phoff) {
        if (g_verbose) fprintf(stderr, "[SIZE_SEGMENTS] ERROR: No program header table (e_phoff=0)\n\n");
        return 0;
    }
    if (!eh->e_phentsize) {
        if (g_verbose) fprintf(stderr, "[SIZE_SEGMENTS] ERROR: Invalid e_phentsize=0\n\n");
        return 0;
    }
    if (eh->e_phentsize != sizeof(Elf64_Phdr)) {
        if (g_verbose) {
            fprintf(stderr, "[SIZE_SEGMENTS] ERROR: e_phentsize=%u mismatch (expect %zu)\n\n",
                    eh->e_phentsize, sizeof(Elf64_Phdr));
        }
        return 0;
    }

    if (g_verbose) {
        fprintf(stderr, "[SIZE_SEGMENTS] Program header table at offset: %" PRIu64 "\n", eh->e_phoff);
        fprintf(stderr, "[SIZE_SEGMENTS] Scanning %u program segments...\n", eh->e_phnum);
    }

    size_t max_end = 0;
    int processed = 0, skipped_null = 0;

    for (uint16_t i = 0; i < eh->e_phnum; ++i) {
        const Elf64_Phdr* p = (const Elf64_Phdr*)(data + eh->e_phoff + (size_t)i * eh->e_phentsize);

        if (p->p_type == 0 /*PT_NULL*/) {
            if (g_verbose) fprintf(stderr, "[SIZE_SEGMENTS] Segment[%u]: SKIP PT_NULL\n", i);
            skipped_null++;
            continue;
        }

        uint64_t off = p->p_offset;
        uint64_t sz  = p->p_filesz;
        if (sz > SIZE_MAX - off) {
            if (g_verbose) {
                fprintf(stderr, "[SIZE_SEGMENTS] ERROR: Segment[%u] overflow: off=%" PRIu64 ", filesz=%" PRIu64 "\n\n",
                        i, off, sz);
            }
            return 0;
        }

        size_t end = (size_t)(off + sz);
        if (end > max_end) {
            if (g_verbose) {
                fprintf(stderr, "[SIZE_SEGMENTS] Segment[%u]: NEW MAX END = %zu\n", i, end);
                fprintf(stderr, "    Type: 0x%X (%s), Offset: %" PRIu64 ", FileSize: %" PRIu64 ", MemSize: %" PRIu64 "\n",
                        p->p_type, ph_type_name(p->p_type), off, sz, p->p_memsz);
                print_ph_flags(p->p_flags);
            }
            max_end = end;
        } else if (g_verbose) {
            fprintf(stderr, "[SIZE_SEGMENTS] Segment[%u]: End = %zu (offset=%" PRIu64 ", filesz=%" PRIu64 ", type=0x%X/%s)\n",
                    i, end, off, sz, p->p_type, ph_type_name(p->p_type));
        }
        processed++;
    }

    if (g_verbose) {
        fprintf(stderr, "[SIZE_SEGMENTS] SUMMARY: Processed %d segments, skipped %d PT_NULL\n",
                processed, skipped_null);
        fprintf(stderr, "[SIZE_SEGMENTS] FINAL SIZE from segments: %zu bytes\n", max_end);
        fprintf(stderr, "=== Segment Size Calculation Complete ===\n\n");
    }
    return max_end;
}

/* ---------------------- public: robust size ---------------------- */
size_t get_cubin_code_size(const void* raw) {
    
    if (!raw) return 0;

    const unsigned char* data = (const unsigned char*)raw;
    /* 先只支持裸 ELF；fatbin 请在上层透传 */
    if (!is_valid_elf64_le(data, sizeof(Elf64_Ehdr))) {
        vlog("[enc] not an ELF64 image; size unknown (will pass-through)\n");
        return 0;
    }
    const Elf64_Ehdr* eh = (const Elf64_Ehdr*)data;

    size_t a = size_from_sections(data, eh);
    size_t b = size_from_segments(data, eh);
    size_t sz = (a > b ? a : b);

    if (g_verbose) {
        fprintf(stderr, "[enc] ELF size via sections=%zu, segments=%zu, choose=%zu\n", a, b, sz);
    }
    return sz;
}

/* ========= 工具：取节名 ========= */
const char* get_sh_name(const unsigned char* base,
                               const Elf64_Ehdr* eh,
                               const Elf64_Shdr* shstr,
                               const Elf64_Shdr* sec) {
    if (!shstr || !sec) return NULL;
    const char* strtab = (const char*)(base + shstr->sh_offset);
    uint32_t off = sec->sh_name;
    return (off < shstr->sh_size) ? (strtab + off) : NULL;
}

/* ========= 工具：是否 .text.* 可执行段 ========= */
bool is_exec_text_section(const char* name, const Elf64_Shdr* s) {
    if (!name) return 0;
    if (s->sh_type != 1 /*SHT_PROGBITS*/) return 0;
    if ((s->sh_flags & 0x4 /*SHF_EXECINSTR*/) == 0) return 0;
    /* 节名以 ".text" 开头（匹配 ".text" 和 ".text.*"） */
    if (name[0]=='.' && name[1]=='t' && name[2]=='e' && name[3]=='x' && name[4]=='t') return 1;
    return 0;
}




/* --------- 查找某个名字的节（如 .symtab / .dynsym / .strtab / .dynstr） --------- */
const Elf64_Shdr* find_section_by_name(const unsigned char* base,
                                              const Elf64_Ehdr* eh,
                                              const char* wanted) {
    if (!eh->e_shoff || !eh->e_shnum) return NULL;
    const Elf64_Shdr* shdr0 = (const Elf64_Shdr*)(base + eh->e_shoff);

    /* 定位节名字符串表（.shstrtab） */
    uint16_t shstr_idx = eh->e_shstrndx;
    if (shstr_idx == 0 || shstr_idx >= eh->e_shnum) return NULL;
    const Elf64_Shdr* shstr = &shdr0[shstr_idx];

    for (uint16_t i = 0; i < eh->e_shnum; ++i) {
        const Elf64_Shdr* s = &shdr0[i];
        const char* name = get_sh_name(base, eh, shstr, s);
        if (name && strcmp(name, wanted) == 0) return s;
    }
    return NULL;
}



std::vector<FuncLen> collect_func_text_lengths(const unsigned char* base, size_t len) {
    std::vector<FuncLen> out;
    if (!is_valid_elf64_le(base, len)) return out;

    const Elf64_Ehdr* eh = (const Elf64_Ehdr*)base;
    if (!eh->e_shoff || !eh->e_shnum) return out;

    const Elf64_Shdr* shdr0 = (const Elf64_Shdr*)(base + eh->e_shoff);
    const Elf64_Shdr* shstr = NULL;
    if (eh->e_shstrndx && eh->e_shstrndx < eh->e_shnum) {
        shstr = &shdr0[eh->e_shstrndx];
    }

    // 先找 .symtab，找不到再用 .dynsym
    const Elf64_Shdr* symtab = find_section_by_name(base, eh, ".symtab");
    const Elf64_Shdr* strtab = NULL;
    if (!symtab) {
        symtab = find_section_by_name(base, eh, ".dynsym");
        if (symtab) strtab = find_section_by_name(base, eh, ".dynstr");
    } else {
        if (symtab->sh_link < eh->e_shnum) strtab = &shdr0[symtab->sh_link];
    }
    if (!symtab || !strtab) return out;

    const Elf64_Sym* syms = (const Elf64_Sym*)(base + symtab->sh_offset);
    size_t sym_cnt = (size_t)(symtab->sh_size / (symtab->sh_entsize ? symtab->sh_entsize : sizeof(Elf64_Sym)));
    const char* str_base = (const char*)(base + strtab->sh_offset);

    for (size_t i = 0; i < sym_cnt; ++i) {
        const Elf64_Sym* sym = &syms[i];
        if (ELF64_ST_TYPE(sym->st_info) != 2 /* STT_FUNC */) continue;
        if (sym->st_shndx == 0 /* SHN_UNDEF */) continue;
        if (sym->st_name >= strtab->sh_size) continue;

        // 所在节必须是 .text 或 .text.*
        if (sym->st_shndx >= eh->e_shnum) continue;
        const Elf64_Shdr* sec = &shdr0[sym->st_shndx];
        const char* secname = get_sh_name(base, eh, shstr, sec);
        if (!is_exec_text_section(secname, sec)) continue;

        // 函数名
        const char* name_mangled = str_base + sym->st_name;
        char dem_buf[1024];

        // 长度 = st_size（注意 strip 过的目标可能为0）
        uint64_t fsz = sym->st_size;

        FuncLen rec;
        rec.name_mangled  = name_mangled ? name_mangled : "";
        rec.text_size     = fsz;
        out.push_back(std::move(rec));
    }

    return out;
}

std::vector<FuncLen> collect_mainfunc_text_lengths(const unsigned char* base, size_t len) {
    std::vector<FuncLen> out;
    if (!is_valid_elf64_le(base, len)) return out;

    const Elf64_Ehdr* eh = (const Elf64_Ehdr*)base;
    if (!eh->e_shoff || !eh->e_shnum) return out;

    const Elf64_Shdr* shdr0 = (const Elf64_Shdr*)(base + eh->e_shoff);
    const Elf64_Shdr* shstr = NULL;
    if (eh->e_shstrndx && eh->e_shstrndx < eh->e_shnum) {
        shstr = &shdr0[eh->e_shstrndx];
    }

    // 先找 .symtab，找不到再用 .dynsym
    const Elf64_Shdr* symtab = find_section_by_name(base, eh, ".symtab");
    const Elf64_Shdr* strtab = NULL;
    if (!symtab) {
        symtab = find_section_by_name(base, eh, ".dynsym");
        if (symtab) strtab = find_section_by_name(base, eh, ".dynstr");
    } else {
        if (symtab->sh_link < eh->e_shnum) strtab = &shdr0[symtab->sh_link];
    }
    if (!symtab || !strtab) return out;

    const Elf64_Sym* syms = (const Elf64_Sym*)(base + symtab->sh_offset);
    size_t sym_cnt = (size_t)(symtab->sh_size / (symtab->sh_entsize ? symtab->sh_entsize : sizeof(Elf64_Sym)));
    const char* str_base = (const char*)(base + strtab->sh_offset);

    for (size_t i = 0; i < sym_cnt; ++i) {
        const Elf64_Sym* sym = &syms[i];
        if (ELF64_ST_TYPE(sym->st_info) != 2 /* STT_FUNC */) continue;
        if (sym->st_shndx == 0 /* SHN_UNDEF */) continue;
        if (sym->st_name >= strtab->sh_size) continue;

        // 所在节必须是 .text 或 .text.*
        if (sym->st_shndx >= eh->e_shnum) continue;
        const Elf64_Shdr* sec = &shdr0[sym->st_shndx];
        const char* secname = get_sh_name(base, eh, shstr, sec);
        if (!is_exec_text_section(secname, sec)) continue;

        // 函数名
        const char* name_mangled = str_base + sym->st_name;
        
        // 过滤条件：排除内联/辅助函数
        // 1. 排除包含 "$" 符号的函数名（内联函数）
        if (strchr(name_mangled, '$') != NULL) {
            continue;
        }
        
        // 2. 可选：排除名称过短的函数（通常是辅助函数）
        // if (strlen(name_mangled) < 5) continue;
        
        // 3. 可选：只保留特定模式的主函数
        // 例如：只保留包含 "kernel" 的函数
        // if (strstr(name_mangled, "kernel") == NULL) continue;

        char dem_buf[1024];

        // 长度 = st_size
        uint64_t fsz = sym->st_size;

        FuncLen rec;
        rec.name_mangled  = name_mangled ? name_mangled : "";
        rec.text_size     = fsz;
        out.push_back(std::move(rec));
    }

    return out;
}

