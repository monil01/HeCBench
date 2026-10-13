IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
RUN_ARGS ?= 100000 100
LAUNCHER ?=
DEBUG ?= no
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
ifeq ($(DEBUG),yes)
CPPFLAGS += -DDEBUG
endif
all: main kernel.hip
main: main.cpp Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp kernels.h Makefile
	$(HIPCC) -O3 --genco $< -o $@
kernel.ptx: kernel.cu kernels.h
	nvcc -O3 -ptx $< -o $@
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
