IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
ROCM_PATH ?= /opt/rocm
CXX ?= g++
HIPCC ?= hipcc
RUN_ARGS ?= 4
LAUNCHER ?=
OPTIMIZE ?= yes
CXXFLAGS += -std=c++17 -Wall -D__HIP_PLATFORM_AMD__ -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include -I$(ROCM_PATH)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64 -ldl -lpthread
ifeq ($(OPTIMIZE),yes)
CXXFLAGS += -O3
HIPFLAGS += -O3
endif
all: main kernel.hip
main: main.cpp Makefile
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ $<
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip *.o
.PHONY: all run clean
