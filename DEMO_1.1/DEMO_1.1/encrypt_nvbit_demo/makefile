# ================== 编译器与工具 ==================
CXX ?= g++
NVCC=nvcc -ccbin=$(CXX) -D_FORCE_INLINES
PTXAS=ptxas
OBJCOPY?=objcopy

# ================== 优化等级（DEBUG=1 开启调试） ==================
ifeq ($(DEBUG),1)
NVCC_OPT=-O0 -g
CXX_OPT=-O0 -g
else
NVCC_OPT=-O3
CXX_OPT=-O3
endif

# ================== 版本检查 ==================
NVCC_VER_REQ=10.1
NVCC_VER=$(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_VER_CHECK=$(shell echo "${NVCC_VER} >= $(NVCC_VER_REQ)" | bc)
ifeq ($(NVCC_VER_CHECK),0)
$(error ERROR: nvcc version >= $(NVCC_VER_REQ) required to compile an nvbit tool! Instrumented applications can still use lower versions of nvcc.)
endif

# ptxas 行为分界版本（是否需要 -maxrregcount 以规避某些寄存器分配问题）
PTXAS_VER_ADD_FLAG=12.3
PTXAS_VER=$(shell $(PTXAS) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
PTXAS_VER_CHECK=$(shell echo "${PTXAS_VER} >= $(PTXAS_VER_ADD_FLAG)" | bc)

# ================== NVBit 路径与库 ==================
NVBIT_PATH=../core
PROJECT_INC_DIR=./include        # ★ 新增：你的头文件目录

INCLUDES=-I$(NVBIT_PATH) -I$(PROJECT_INC_DIR)
LIBS=-L$(NVBIT_PATH) -lnvbit

# 从 nvcc 的路径推导出 CUDA lib64，用于 -lcuda
NVCC_PATH=-L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))


# ================== FATBIN 相关（把设备端 helper 编到 fatbin 并嵌入） ==================
FATBIN_CU := decrypt_kernel/decrypt_kernel.cu
FATBIN_BIN := fatbin/decrypt_bytes_kernel.fatbin
FATBIN_OBJ := fatbin/decrypt_bytes_kernel_fatbin.o

UNAME_M:=$(shell uname -m)
ifeq ($(UNAME_M),x86_64)
  OBJCOPY_O=elf64-x86-64
  BFD=i386:x86-64
else ifeq ($(UNAME_M),aarch64)
  OBJCOPY_O=elf64-aarch64
  BFD=aarch64
else
  OBJCOPY_O=elf64-x86-64
  BFD=i386:x86-64
endif

# ================== 源与目标 ==================
# 仅收集主机端 .cu（排除 FATBIN_CU）
ALL_CU := $(wildcard src/*.cu)
CU_SOURCES := $(filter-out $(FATBIN_CU),$(ALL_CU))
CU_OBJECTS := $(CU_SOURCES:.cu=.o)

# 收集主机端 .cpp（例如 helper_decrypt_kernel.cpp / globals.cc 等）
CPP_SOURCES := $(wildcard src/*.cpp)
CPP_OBJECTS := $(CPP_SOURCES:.cpp=.o)

# 统一的目标
OBJECTS := $(CU_OBJECTS) $(CPP_OBJECTS)

# 可配置的 GPU 架构
ARCH?=sm_86

# 生成的 NVBit 工具名（按当前目录名命名 .so）
mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))
NVBIT_TOOL=$(current_dir).so

# ================== 规则 ==================
.PHONY: all clean fatbin-dir print
print:
	@echo "=== Makefile Debug Info ==="
	@echo "PWD:            $(shell pwd)"
	@echo "ARCH:           $(ARCH)"
	@echo "ALL_CU:         $(ALL_CU)"
	@echo "FATBIN_CU:      $(FATBIN_CU)"
	@echo "CU_SOURCES:     $(CU_SOURCES)"
	@echo "CPP_SOURCES:    $(CPP_SOURCES)"
	@echo "CU_OBJECTS:     $(CU_OBJECTS)"
	@echo "CPP_OBJECTS:    $(CPP_OBJECTS)"
	@echo "OBJECTS:        $(OBJECTS)"
	@echo "NVBIT_PATH:     $(NVBIT_PATH)"
	@echo "NVCC_PATH:      $(NVCC_PATH)"


all: $(NVBIT_TOOL)

# 最终 .so：将主机端对象 + 嵌入的 fatbin 对象 + NVBit + libcuda 链接在一起
$(NVBIT_TOOL): $(OBJECTS) $(FATBIN_OBJ) $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) $(NVCC_OPT) \
	    $(OBJECTS) $(FATBIN_OBJ) \
	    $(LIBS) $(NVCC_PATH) -lcuda \
	    -shared -o $@

# -------- 编译主机端 .cu -> .o --------
%.o: %.cu
	$(NVCC) -dc -c -std=c++17 $(INCLUDES) -Xptxas -cloning=no -Xcompiler -Wall -arch=$(ARCH) $(NVCC_OPT) -Xcompiler -fPIC $< -o $@

# -------- 编译主机端 .cpp -> .o --------
%.o: %.cpp
	$(CXX) -std=c++17 $(CXX_OPT) -fPIC $(INCLUDES) -Wall -c $< -o $@

# ====== 设备端 helper 生成 fatbin，并用 objcopy 封装成可链接对象 ======
fatbin-dir:
	@mkdir -p fatbin

# 1) decrypt_kernel.cu -> fatbin/decrypt_bytes_kernel.fatbin
$(FATBIN_BIN): $(FATBIN_CU) | fatbin-dir
	$(NVCC) -fatbin -arch=$(ARCH) -O2 $< -o $@

# 2) .fatbin -> fatbin/decrypt_bytes_kernel_fatbin.o
$(FATBIN_OBJ): $(FATBIN_BIN)
	$(OBJCOPY) -I binary -O $(OBJCOPY_O) -B $(BFD) \
	  --redefine-sym _binary_fatbin_decrypt_bytes_kernel_fatbin_start=_binary_decrypt_bytes_kernel_fatbin_start \
	  --redefine-sym _binary_fatbin_decrypt_bytes_kernel_fatbin_end=_binary_decrypt_bytes_kernel_fatbin_end \
	  --redefine-sym _binary_fatbin_decrypt_bytes_kernel_fatbin_size=_binary_decrypt_bytes_kernel_fatbin_size \
	  $< $@

clean:
	rm -f *.so *.o *.fatbin
	rm -f fatbin/*.fatbin fatbin/*.o
	rm -f src/*.o
	
