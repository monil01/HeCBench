IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX = g++
HIPCC ?= hipcc
RUN_ARGS ?=
LAUNCHER ?=
all: main kernel.hip
main: main.cpp KeccakF.cpp KeccakTreeCPU.cpp KeccakTreeGPU.cpp Test.cpp $(wildcard *.h)
	$(CXX) -O3 -std=c++17 -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include main.cpp KeccakF.cpp KeccakTreeCPU.cpp KeccakTreeGPU.cpp Test.cpp -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -lpthread -ldl
kernel.hip: kernel.hip.cpp KeccakTree.h KeccakTypes.h KeccakF.h
	$(HIPCC) -O3 --genco kernel.hip.cpp -o $@
kernel.ptx: kernel.cu KeccakTree.h KeccakTypes.h KeccakF.h
	nvcc -O3 -ptx kernel.cu -o $@
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
