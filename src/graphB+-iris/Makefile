IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
ROCM_PATH ?= /opt/rocm
CXX ?= g++
HIPCC ?= hipcc
VERIFY ?= no
RUN_ARGS ?= graph.csv 10000 output
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -D__HIP_PLATFORM_AMD__ -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include -I$(ROCM_PATH)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib
LDLIBS += -liris -lamdhip64 -ldl -lpthread
ifeq ($(VERIFY),yes)
CPPFLAGS += -DVERIFY
HIPFLAGS += -DVERIFY
endif
all: main kernel.hip
main: main.cpp graph.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ main.cpp $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) $(HIPFLAGS) -O3 --genco -o $@ kernel.hip.cpp
kernel.ptx: kernel.cu
	nvcc $(HIPFLAGS) -O3 -ptx -o $@ kernel.cu
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean

