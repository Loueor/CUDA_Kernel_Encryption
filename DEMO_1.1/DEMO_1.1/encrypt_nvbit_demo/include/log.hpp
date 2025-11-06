#pragma once
#include <cstdarg>
#include <cstdio>

extern int g_verbose;


void init_verbose_once();
void vlog(const char* fmt, ...);
