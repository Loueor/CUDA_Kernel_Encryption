#include "log.hpp"
#include <cstdlib>

int g_verbose = 0;

void init_verbose_once() {
    static int inited = 0;
    if (inited) return;
    inited = 1;
    const char* v = std::getenv("ENC_VERBOSE");
    g_verbose = (v && *v && *v != '0') ? 1 : 0;
}

void vlog(const char* fmt, ...) {
    if (!g_verbose) return;
    va_list ap; va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}
