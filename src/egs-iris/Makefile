IRIS_ROOT ?= /tmp/iris
HIP_ROOT ?= /opt/rocm
CXX = g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
EXTRA_CFLAGS ?=
CPPFLAGS += $(EXTRA_CFLAGS) -I$(IRIS_ROOT)/include -I$(IRIS_ROOT)/build/include -I$(HIP_ROOT)/include
LDFLAGS += -L$(IRIS_ROOT)/build/src/runtime -Wl,-rpath,$(IRIS_ROOT)/build/src/runtime -liris
RUN_ARGS ?=
all: main kernel.hip
main: main.cpp EGS.h media.c media_types.h init.cpp output.c bmp.c dmem.h binding_host.h binding_types.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) main.cpp -o $@ $(LDFLAGS)
kernel.hip: kernel.hip.cpp EGS.h media_types.h kernels_hip.inc binding_gpu.h binding_types.h
	$(HIPCC) -O3 $(EXTRA_CFLAGS) --genco $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
