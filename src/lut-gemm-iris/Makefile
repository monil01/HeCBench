IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
HIPCC ?= hipcc
RUN_ARGS ?= 1 100
CXXFLAGS ?= -O3 -std=c++17
all: main kernel.hip libvendor.so
main: main.cpp vendor.cpp Makefile
	$(HIPCC) $(CXXFLAGS) -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include main.cpp -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread -lhipblas
kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) -O3 --genco $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip libvendor.so *.o

libvendor.so: bridge.cpp ../../tools/iris_vendor_bridge.cpp ../../tools/iris_vendor_bridge.hpp
	$(HIPCC) -O3 -std=c++17 -fPIC -shared bridge.cpp -o $@
