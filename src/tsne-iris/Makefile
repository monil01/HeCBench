IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
ROCM_PATH ?= /opt/rocm
CXX ?= g++
HIPCC ?= hipcc
RUN_ARGS ?=
SOURCES := $(wildcard native/*.cu)
OBJECTS := $(SOURCES:.cu=.o)
HEADERS := $(wildcard native/*.h native/*.hpp) dmem_adapter.hpp
COMMON := -O3 -ffast-math -std=c++17 -fPIC -D__HIP_PLATFORM_AMD__ -I. -I$(ROCM_PATH)/include -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
HIPFLAGS += $(COMMON)
LIBS := -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64 -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
all: main libhecbench_vendor.so
native/%.o: native/%.cu $(HEADERS)
	$(HIPCC) $(HIPFLAGS) -c $< -o $@
dmem_adapter.o: dmem_adapter.cpp dmem_adapter.hpp
	$(HIPCC) $(HIPFLAGS) -c $< -o $@
libtsne.so: $(OBJECTS) dmem_adapter.o
	$(HIPCC) -shared $^ -o $@ $(LIBS)
main: main.cpp libtsne.so
	$(CXX) $(COMMON) $< -L. -Wl,-rpath,'$$ORIGIN' -ltsne -o $@ $(LIBS)
libhecbench_vendor.so: kernel.hip.cpp ../../tools/iris_vendor_bridge.cpp
	$(CXX) -std=c++17 -fPIC -shared $< -o $@
data:
	ln -s ../tsne-hip/data data
run: all data
	HIP_VISIBLE_DEVICES=0 IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main *.o *.so $(OBJECTS)
.PHONY: all run clean
