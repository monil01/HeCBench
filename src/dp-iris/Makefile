IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
HIPCC ?= hipcc
HIP_ARCH ?= native
RUN_ARGS ?= 268435456 1000
CXXFLAGS ?= -O3 -std=c++17
all: main kernel.hip libhecbench_vendor.so
main: main.cpp ../../tools/iris_vendor_bridge.hpp Makefile
	$(HIPCC) $(CXXFLAGS) --offload-arch=$(HIP_ARCH) -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include main.cpp -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -lhipblas -ldl -lpthread
kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) -O3 -munsafe-fp-atomics --offload-arch=$(HIP_ARCH) --genco $< -o $@
libhecbench_vendor.so: bridge.cpp ../../tools/iris_vendor_bridge.cpp ../../tools/iris_vendor_bridge.hpp
	$(HIPCC) -O2 -std=c++17 -fPIC -shared $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip libhecbench_vendor.so *.o
