# 简单 Makefile（请按实际 CUDA 安装位置设置 NVCC/ARCH）
NVCC ?= nvcc
ARCH ?= sm_86
CXXFLAGS := -O3 -std=c++14

all: testapp

testapp: testapp.cu
	$(NVCC) $(CXXFLAGS) -arch=$(ARCH) testapp.cu -o testapp

clean:
	rm -f testapp
