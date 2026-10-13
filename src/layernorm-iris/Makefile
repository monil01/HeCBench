IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX = hipcc
HIPCC ?= hipcc
RUN_ARGS ?=
DEFAULT_RUN_ARGS_1 ?= 8 1024 768 2000
DEFAULT_RUN_ARGS_2 ?= 8 1024 12288 2000
LAUNCHER ?=
EXTRA_CFLAGS ?=
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include $(EXTRA_CFLAGS)
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
all: main kernel.hip
main: main.cpp dmem_runtime.hpp Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp kernels.h Makefile
	$(HIPCC) -O3 -std=c++17 $(EXTRA_CFLAGS) --genco $< -o $@
kernel.ptx: kernel.cu kernels.h
	nvcc -O3 -ptx $< -o $@
run: all
ifeq ($(strip $(RUN_ARGS)),)
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(DEFAULT_RUN_ARGS_1)
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(DEFAULT_RUN_ARGS_2)
else
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
endif
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
