IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
LDIM ?= 32
ITERATIONS ?= 100
PRECISION ?= 1
DEFINES = -DLDIM=$(LDIM) -DITERATIONS=$(ITERATIONS) -DPRECISION=$(PRECISION)
HIPFLAGS ?= -O3 -ffast-math
RUN_ARGS ?= 256
#
all: main kernel.hip
main: main.cpp dslash.cpp reference.cpp dslash.h Makefile
	$(CXX) $(CXXFLAGS) $(DEFINES) -fopenmp -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include main.cpp dslash.cpp reference.cpp -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
kernel.hip: kernel.hip.cpp dslash.h Makefile
	$(HIPCC) $(HIPFLAGS) $(DEFINES) --genco $< -o $@
kernel.ptx: kernel.cu
	nvcc -O3 -ptx $< -o $@
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
