IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include -I../boxfilter-sycl
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
RUN_ARGS ?=
all: main kernel.hip
main: main.cpp dmem.h permutations.h dds.h Makefile ../boxfilter-sycl/shrUtils.cpp ../boxfilter-sycl/shrUtils.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.cpp ../boxfilter-sycl/shrUtils.cpp -o $@ $(LDFLAGS)
kernel.hip: kernel.hip.cpp
	$(HIPCC) -O3 --genco $< -o $@
kernel.ptx: kernel.cu helper_math.h
	nvcc -O3 -ptx $< -o $@
run: all
	@if [ -n "$(RUN_ARGS)" ]; then IRIS_ARCHS=HIP ./main $(RUN_ARGS); else IRIS_ARCHS=HIP ./main ../dxtc2-sycl/data/lena_std.ppm ../dxtc2-sycl/data/lena_ref.dds 100 && IRIS_ARCHS=HIP ./main ../dxtc2-sycl/data/teapot512_std.ppm ../dxtc2-sycl/data/teapot512_ref.dds 100; fi
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
