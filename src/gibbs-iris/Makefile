IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX = hipcc
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Wall
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
ROCM_PATH ?= /opt/rocm
CPPFLAGS += -I$(ROCM_PATH)/include/hiprand -I$(ROCM_PATH)/include/rocrand
HIPFLAGS ?= -O3 -I$(ROCM_PATH)/include/hiprand -I$(ROCM_PATH)/include/rocrand
LAUNCHER ?=
RUN_ARGS ?= ../gibbs-cuda/data.csv 5000
OUTPUT ?= no
ifeq ($(OUTPUT),yes)
CPPFLAGS += -DBENCH_PRINT
endif
.PHONY: all clean run
all: main kernel.hip
main: main.cpp iris_support.hpp Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp kernels.hpp Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ $<
kernel.ptx: kernel.cu kernels.hpp
	nvcc -O3 -ptx -o $@ $<
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
