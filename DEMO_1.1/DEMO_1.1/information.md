GPU Memory Architecture on Ampere (RTX 30-Series and Newer)

NVIDIA Ampere GPUs implement a virtual memory system similar in spirit to CPUs. Each GPU process (CUDA context) gets its own page table for translating device virtual addresses to physical memory addresses
usenix.org
. Notably, Ampere and newer use a 5-level page table format to accommodate large address spaces (up to 49-bit virtual addresses)
usenix.org
nvidia.github.io
. Page tables are managed by the GPU driver and stored in GPU memory (with a mirror copy in host memory for the driver’s use)
usenix.org
. The GPU’s Memory Management Unit (GMMU) walks these page tables on-demand, caching translations in TLBs (each SM has an L1 TLB and there’s a shared L2 TLB)
usenix.org
. All page table levels and entries are 4 KB-aligned, and Ampere supports standard 4KB pages (and larger “big pages” like 2MB)
nvidia.github.io
.

Crucially, code and data reside in the same physical memory (device DRAM) and share the same address space from the GPU’s perspective. When you launch a CUDA kernel, the compiled machine code (SASS instructions) for that kernel is loaded into the device’s memory (often into a region of global memory) by the driver. At runtime, the GPU’s Streaming Multiprocessors (SMs) fetch instructions from this memory through an instruction cache hierarchy. For example, an Ampere SM has an L1 instruction cache (on the order of a few KB per SM), backed by the L2 cache and device DRAM
forums.developer.nvidia.com
. If the kernel’s code is small, it may entirely sit in the instruction cache; if not, the GPU will fetch additional instructions from device memory on cache misses
forums.developer.nvidia.com
. There is no dedicated “code memory” region – kernels “are stored in global memory” when not in cache
forums.developer.nvidia.com
. This means that from the hardware’s view, a pointer to an instruction in a kernel is essentially a pointer to a location in global device memory. In summary, Ampere GPUs use unified device memory for both code and data, with a common page table translating addresses for both.

GPU Page Table Structure and Permissions: Each page table entry (PTE) on modern NVIDIA GPUs contains physical address bits and some flags, but notably does not contain an execute-disable or “NX” bit. In fact, NVIDIA’s published page table format for Pascal/Ampere has no explicit executable or no-execute flag
usenix.org
. The absence of an execute permission bit means the GPU does not distinguish whether a page holds data or instructions – if an SM’s program counter is pointed to an address, the hardware will attempt to execute whatever bytes are there
usenix.org
. Similarly, there is a “read-only” bit in the PTE format (and a “dirty” bit for tracking writes), but in practice these are either unused or always unset by current drivers
usenix.org
. Researchers confirmed that on Ampere GPUs the read-only flag remains 0 even for actual code pages, indicating they were mapped as writable in the page table
usenix.org
. (There is also a “privilege” bit in the PTE intended to mark supervisor-only pages, but it appears to be unused or ignored in user CUDA contexts
nvidia.github.io
.) Overall, the GPU’s memory management does not enforce the typical CPU protections like user/kernel mode or write-xor-execute (W^X) at the hardware level. All pages that a CUDA program can access are, by design, both readable/writable and executable from the GPU’s standpoint
usenix.org
. This design, while flexible for programming, has significant security implications as we’ll see.