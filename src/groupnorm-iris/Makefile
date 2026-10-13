IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
RUN_ARGS ?=
all: main kernel.hip
main: main.cpp reference.h common.h Makefile
	$(CXX) $(CXXFLAGS) -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include $< -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
kernel.hip: kernel.hip.cpp common.h Makefile
	$(HIPCC) -O3 --genco $< -o $@
kernel.ptx: kernel.cu
	nvcc -O3 -ptx $< -o $@
run: all
	@if [ -n "$(RUN_ARGS)" ]; then IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS); else IRIS_ARCHS=HIP $(LAUNCHER) ./main 32 32 16 16 32 100 && IRIS_ARCHS=HIP $(LAUNCHER) ./main 32 128 64 64 32 100; fi
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
