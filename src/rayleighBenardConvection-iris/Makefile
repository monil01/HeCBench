IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
HIPCC ?= hipcc
RUN_ARGS ?= 1000
CXXFLAGS ?= -O3 -std=c++17
all: main kernel.hip libhecbench_vendor.so
main: main.cpp dmem_ops.hpp parameters.hpp ../../tools/iris_vendor_bridge.hpp Makefile
	$(HIPCC) $(CXXFLAGS) -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include main.cpp -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread -lhipblas
kernel.hip: kernel.hip.cpp parameters.hpp Makefile
	$(HIPCC) -O3 --genco $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip libhecbench_vendor.so *.o

libhecbench_vendor.so: bridge.cpp ../../tools/iris_vendor_bridge.cpp ../../tools/iris_vendor_bridge.hpp
	$(HIPCC) -O2 -std=c++17 -shared -fPIC bridge.cpp -o $@
