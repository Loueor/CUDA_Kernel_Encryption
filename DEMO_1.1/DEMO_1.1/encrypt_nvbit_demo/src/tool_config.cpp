#include "tool_config.hpp"
#include <cstdlib>
#include <string>

uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval   = 0xFFFFFFFFu;
int verbose = 0;
int mangled = false;

static int get_env_int(const char* k, int defv) {
    const char* v = std::getenv(k);
    if (!v || !*v) return defv;
    return std::atoi(v);
}

void load_tool_env() {
    instr_begin_interval = (uint32_t)get_env_int("INSTR_BEGIN", 0);
    instr_end_interval   = (uint32_t)get_env_int("INSTR_END", 0xFFFFFFFFu);
    mangled = get_env_int("MANGLED_NAMES", 1) != 0;
    verbose = get_env_int("TOOL_VERBOSE", 0);
}
