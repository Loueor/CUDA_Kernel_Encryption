#include "func_index.hpp"
#include <vector>
#include <mutex>

namespace {
    std::vector<FuncLen> g_funcs;
    std::mutex g_mu;
}

namespace func_index {
    void append_and_take_ownership(std::vector<FuncLen>&& fv) {
        std::lock_guard<std::mutex> lk(g_mu);
        g_funcs.insert(g_funcs.end(),
                       std::make_move_iterator(fv.begin()),
                       std::make_move_iterator(fv.end()));
    }
    uint64_t get_and_remove_func_len(const std::string& kname) {
        std::lock_guard<std::mutex> lk(g_mu);
        for (auto it = g_funcs.begin(); it != g_funcs.end(); ++it) {
            if (it->name_mangled == kname) {
                uint64_t len = it->text_size;
                g_funcs.erase(it);
                return len;
            }
        }
        return 0;
    }
    void clear() {
        std::lock_guard<std::mutex> lk(g_mu);
        g_funcs.clear();
    }
}
