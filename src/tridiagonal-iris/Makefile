IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Wall -DUNIX
BENCH_ITERATIONS ?= 100
CPPFLAGS += -DBENCH_ITERATIONS=$(BENCH_ITERATIONS)
HIPFLAGS ?= -O3 -munsafe-fp-atomics
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
RUN_ARGS ?= -num_systems=524288
LAUNCHER ?=
all: main kernel.hip
main: main.cpp shrUtils.cpp cmd_arg_reader.cpp task_launch.h pcr_small_systems.h cyclic_small_systems.h sweep_small_systems.h test_gen_result_check.h tridiagonal.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ main.cpp shrUtils.cpp cmd_arg_reader.cpp $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp pcr_kernels.cu cyclic_kernels.cu sweep_kernels.cu tridiagonal.h Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ kernel.hip.cpp
kernel.ptx: kernel.cu
	nvcc -std=c++17 -ptx -o $@ $<
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
	IRIS_ARCHS=HIP $(LAUNCHER) ./main --lmem $(RUN_ARGS)
	IRIS_ARCHS=HIP $(LAUNCHER) ./main --vec4 $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
