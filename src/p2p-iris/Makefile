IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
ROCM_PATH ?= /opt/rocm
CXX ?= g++
HIPCC ?= hipcc
OPTIMIZE ?= yes
LAUNCHER ?=
RUN_ARGS ?= 2000
EXTRA_CXXFLAGS ?=
CXXFLAGS += -std=c++17 -Wall -D__HIP_PLATFORM_AMD__ -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include -I$(ROCM_PATH)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64 -ldl -lpthread
ifeq ($(OPTIMIZE),yes)
CXXFLAGS += -O3
HIPFLAGS += -O3
endif
all: main kernel.hip
main: main.cpp Makefile
	$(CXX) $(CXXFLAGS) $(EXTRA_CXXFLAGS) -o $@ $< $(LDFLAGS)
kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ $<
runtime_test: runtime_test.cpp Makefile
	$(CXX) $(CXXFLAGS) $(EXTRA_CXXFLAGS) -o $@ $< $(LDFLAGS)
run: all
	HSA_FORCE_FINE_GRAIN_PCIE=1 IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip runtime_test *.o
.PHONY: all run clean
