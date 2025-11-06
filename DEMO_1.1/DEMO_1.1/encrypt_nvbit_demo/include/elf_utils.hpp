#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>


#ifndef ELF64_ST_TYPE
#define ELF64_ST_TYPE(i)   ((i)&0xF)
#endif
struct FuncLen {
    std::string name_mangled;
    uint64_t    text_size;
};

typedef struct {
    uint32_t st_name;
    unsigned char st_info;
    unsigned char st_other;
    uint16_t st_shndx;
    uint64_t st_value;
    uint64_t st_size;
} Elf64_Sym;

/* ---------------------- ELF64 basic structs ---------------------- */
typedef struct {
    unsigned char e_ident[16]; //magic number
    uint16_t e_type; // file type
    uint16_t e_machine; // machine architecher
    uint32_t e_version; //ELF version
    uint64_t e_entry;  // progrom entry point
    uint64_t e_phoff;   // program header offset
    uint64_t e_shoff;   // section header offset
    uint32_t e_flags;   // elf flag
    uint16_t e_ehsize;  //ELF header size
    uint16_t e_phentsize; //program header entry size
    uint16_t e_phnum; //program header entry number
    uint16_t e_shentsize; // section header entry size
    uint16_t e_shnum; // section header entry number
    uint16_t e_shstrndx;  //section string index
} Elf64_Ehdr;

typedef struct {
    uint32_t sh_name;      // Section name index in .shstrtab string table
    uint32_t sh_type;      // Section type (code, data, etc.)
    uint64_t sh_flags;     // Section attributes (writable, executable, etc.)
    uint64_t sh_addr;      // Section virtual address at execution
    uint64_t sh_offset;    // Section file offset
    uint64_t sh_size;      // Section size in bytes
    uint32_t sh_link;      // Link to another section (depends on type)
    uint32_t sh_info;      // Additional section information
    uint64_t sh_addralign; // Section alignment requirement
    uint64_t sh_entsize;   // Entry size if section contains fixed-size entries
} Elf64_Shdr;

typedef struct {
    uint32_t p_type;   // Segment type: what kind of segment this is
    uint32_t p_flags;  // Segment flags: read/write/execute permissions
    uint64_t p_offset; // Segment file offset: where in the file this segment starts
    uint64_t p_vaddr;  // Segment virtual address: where to load in memory
    uint64_t p_paddr;  // Segment physical address: for systems without MMU (usually same as vaddr)
    uint64_t p_filesz; // Segment size in file: how many bytes in the file
    uint64_t p_memsz;  // Segment size in memory: how many bytes in memory (can be larger than filesz)
    uint64_t p_align;  // Segment alignment: memory alignment requirement
} Elf64_Phdr;

const char* sh_type_name(uint32_t t);
void print_sh_flags(uint64_t flg);
const char* ph_type_name(uint32_t t);
void print_ph_flags(uint32_t flg);

bool    is_valid_elf64_le(const unsigned char* data, size_t min_len);
size_t  get_cubin_code_size(const void* raw);

const char* get_sh_name(const unsigned char* base, const Elf64_Ehdr* eh,
                        const Elf64_Shdr* shstr, const Elf64_Shdr* sec);
bool    is_exec_text_section(const char* name, const Elf64_Shdr* s);
const Elf64_Shdr* find_section_by_name(const unsigned char* base, const Elf64_Ehdr* eh, const char* wanted);

std::vector<FuncLen> collect_func_text_lengths(const unsigned char* base, size_t len);
std::vector<FuncLen> collect_mainfunc_text_lengths(const unsigned char* base, size_t len);
