IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX = hipcc
HIPCC ?= hipcc
RUN_ARGS ?= -i ../hbc-cuda/graphs/breast_coexpress.txt -v --printscores=output.txt
LAUNCHER ?=
EXTRA_CFLAGS ?=
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include $(EXTRA_CFLAGS)
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
all: main kernel.hip
main: main.cpp host_kernels.cpp util.cpp parse.cpp sequential.cpp dmem_runtime.hpp Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< host_kernels.cpp util.cpp parse.cpp sequential.cpp -o $@ $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp kernels.h Makefile
	$(HIPCC) -O3 -std=c++17 $(EXTRA_CFLAGS) --genco $< -o $@
kernel.ptx: kernel.cu kernels.h
	nvcc -O3 -ptx $< -o $@
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
