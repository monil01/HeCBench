IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX = hipcc
HIPCC ?= hipcc
RUN_ARGS ?= ../hogbom-cuda/data/dirty_4096.img ../hogbom-cuda/data/psf_4096.img 1000
LAUNCHER ?=
EXTRA_CFLAGS ?=
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include $(EXTRA_CFLAGS) -DVERIFY
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
all: main kernel.hip
main: main.cpp dmem_runtime.hpp Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp device_kernels.h Makefile
	$(HIPCC) -O3 -std=c++17 $(EXTRA_CFLAGS) --genco $< -o $@
kernel.ptx: kernel.cu kernels.h
	nvcc -O3 -ptx $< -o $@
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
