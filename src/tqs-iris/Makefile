IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
RUN_ARGS ?= -f ../tqs-cuda/input/patternsNP100NB512FB25.txt
#
all: main kernel.hip
main: main.cpp host_task.cpp kernel.h $(wildcard support/*.h) Makefile
	$(CXX) $(CXXFLAGS) -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include main.cpp host_task.cpp -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
kernel.hip: kernel.hip.cpp support/task.h Makefile
	$(HIPCC) -O3 --genco $< -o $@
kernel.ptx: kernel.cu
	nvcc -O3 -ptx $< -o $@
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
