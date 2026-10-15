IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX = hipcc
CXXFLAGS ?= -O3 -std=c++17 -Wall
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -lhipfft -lrocfft
RUN_ARGS ?= 1024 1000
all: main libhecbench_vendor.so
main: main.cpp ../slit-cuda/reference.h ../../tools/iris_vendor_bridge.hpp Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
libhecbench_vendor.so: vendor.cpp ../../tools/iris_vendor_bridge.cpp ../../tools/iris_vendor_bridge.hpp
	$(CXX) -std=c++17 -O2 -fPIC -shared -o $@ $<
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main libhecbench_vendor.so *.o
.PHONY: all run clean
