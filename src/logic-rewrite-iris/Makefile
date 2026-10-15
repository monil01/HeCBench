IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
HIPCC ?= hipcc
CXX ?= g++
RUN_ARGS ?= -c "read ../logic-rewrite-cuda/benchmarks/arithmetic/hyp.aig; resyn2; write hyp_resyn2.aig"
SOURCES := $(wildcard native/*.cu)
OBJECTS := $(SOURCES:.cu=.o)
HEADERS := $(wildcard native/*.h native/*.hpp native/*.cuh native/*.inc)
FLAGS := -O3 -std=c++17 -fPIC -ffast-math -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 -I. -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LIBS := -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -lpthread
all: main libhecbench_vendor.so
native/%.o: native/%.cu dmem_adapter.hpp $(HEADERS)
	$(HIPCC) $(FLAGS) -c $< -o $@
dmem_adapter.o: dmem_adapter.cpp dmem_adapter.hpp
	$(HIPCC) $(FLAGS) -c $< -o $@
liblogic.so: $(OBJECTS) dmem_adapter.o
	$(HIPCC) -shared $^ -o $@ $(LIBS)
main: main.cpp liblogic.so
	$(HIPCC) $(FLAGS) $< -L. -Wl,-rpath,'$$ORIGIN' -llogic -o $@ $(LIBS)
libhecbench_vendor.so: kernel.hip.cpp ../../tools/iris_vendor_bridge.cpp
	$(CXX) -O2 -std=c++17 -shared -fPIC $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main *.so *.o native/*.o
.PHONY: all run clean
