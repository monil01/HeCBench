IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Wall
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
HIPFLAGS ?= -O3
LAUNCHER ?=
RUN_ARGS ?= 100 ./Datasets/ERR240727_1_E2_30000Pairs.txt 30000 1000
OUTPUT ?= no
ifeq ($(OUTPUT),yes)
CPPFLAGS += -DBENCH_PRINT
endif
.PHONY: all clean run
all: main kernel.hip
main: main.cpp iris_support.hpp bit_helpers.hpp reference.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp kernels.hpp bit_helpers.hpp Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ $<
kernel.ptx: kernel.cu kernels.hpp
	nvcc -O3 -ptx -o $@ $<
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o

