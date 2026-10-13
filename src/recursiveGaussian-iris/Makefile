IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include -I../boxfilter-sycl
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
RUN_ARGS ?= StoneRGB.ppm 1000
all: main kernel.hip
main: main.cpp RecursiveGaussianHost.cpp main.h dmem.h Makefile ../boxfilter-sycl/shrUtils.cpp ../boxfilter-sycl/shrUtils.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.cpp RecursiveGaussianHost.cpp ../boxfilter-sycl/shrUtils.cpp -o $@ $(LDFLAGS)
kernel.hip: kernel.hip.cpp main.h
	$(HIPCC) -O3 --genco $< -o $@
kernel.ptx: kernel.cu main.h helper_math.h
	nvcc -O3 -ptx $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
