IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
ROCM_PATH ?= /opt/rocm
CXX ?= g++
HIPCC ?= hipcc
OPTIMIZE ?= yes
LAUNCHER ?=
RUN_ARGS ?= 300
EXTRA_CXXFLAGS ?=
CXXFLAGS += -std=c++17 -Wall -fopenmp -D__HIP_PLATFORM_AMD__ -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include -I$(ROCM_PATH)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64 -ldl -lpthread -fopenmp
ifeq ($(OPTIMIZE),yes)
CXXFLAGS += -O3
HIPFLAGS += -O3
endif
all: main kernel.hip
main: main.cpp reference.h Makefile
	$(CXX) $(CXXFLAGS) $(EXTRA_CXXFLAGS) -o $@ $< $(LDFLAGS)
kernel.hip: kernel.hip.cpp kernels.h Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ $<
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip *.o
.PHONY: all run clean
