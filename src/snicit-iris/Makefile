IRIS_ROOT ?= /tmp/iris
IRIS_LIBDIR ?= $(IRIS_ROOT)/build/src/runtime
CXX = hipcc
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
HIPFLAGS ?= -O3 -std=c++17
CLI11_DIR ?= ../snicit-cuda/3rd-party
RUN_ARGS ?= -p ../snicit-cuda/dataset
.PHONY: all run clean
all: main kernel.hip
main: main.cpp SNICIT.hpp dmem.hpp
	$(CXX) $(CXXFLAGS) -I$(CLI11_DIR) -I$(IRIS_ROOT)/include main.cpp -o $@ -L$(IRIS_LIBDIR) -Wl,-rpath,$(IRIS_LIBDIR) -liris
kernel.hip: kernel.hip.cpp
	$(HIPCC) $(HIPFLAGS) --genco $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
