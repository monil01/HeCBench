IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Wall
HIPFLAGS ?= -O3 -munsafe-fp-atomics
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
RUN_ARGS ?=
ifeq ($(VALIDATION),yes)
CPPFLAGS += -DMCW=2 -DMAX_SIM=2 -DMIN_CODEWORD=0
HIPFLAGS += -DMCW=2 -DMAX_SIM=2 -DMIN_CODEWORD=0
endif
LAUNCHER ?=
all: main kernel.hip
main: main.cpp cpu.cpp LDPC.h matrix.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ main.cpp cpu.cpp $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp LDPC.h Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ kernel.hip.cpp
kernel.ptx: kernel.cu
	nvcc -std=c++17 -ptx -o $@ $<
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
