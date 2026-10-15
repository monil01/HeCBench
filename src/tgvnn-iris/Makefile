IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
HIPCC ?= hipcc
HIP_ARCH ?= native
CXXFLAGS ?= -O3 -std=c++17
RUN_ARGS ?=
all: main kernel.hip libhecbench_vendor.so
main: main.cpp ra.cpp ra.h float2math.h dmem_ops.hpp launches.hpp observe.hpp Makefile ../../tools/iris_vendor_bridge.hpp
	$(HIPCC) $(CXXFLAGS) -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include main.cpp ra.cpp -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -lhipfft -lhipblas -lhipsolver -ldl -lpthread
kernel.hip: kernel.hip.cpp float2math.h Makefile
	$(HIPCC) -O3 --offload-arch=$(HIP_ARCH) --genco $< -o $@
libhecbench_vendor.so: bridge.cpp ../../tools/iris_vendor_bridge.cpp ../../tools/iris_vendor_bridge.hpp
	$(HIPCC) -O2 -std=c++17 -fPIC -shared $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip libhecbench_vendor.so *.o
