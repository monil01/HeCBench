IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
HIPCC ?= hipcc
RUN_ARGS ?= -v 10 gisette_scale
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wno-unused-result
all: main libhecbench_vendor.so
main: main.cpp svm.cpp svm.h driver.c ../../tools/iris_vendor_bridge.hpp Makefile
	$(HIPCC) $(CXXFLAGS) -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include main.cpp svm.cpp -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -lhipblas -ldl -lpthread
libhecbench_vendor.so: kernel.hip.cpp ../../tools/iris_vendor_bridge.cpp ../../tools/iris_vendor_bridge.hpp
	$(HIPCC) -std=c++17 -O2 -fPIC -shared $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main libhecbench_vendor.so *.o
