IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Wall
HIPFLAGS ?= -O3 -munsafe-fp-atomics
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
RUN_ARGS ?= --warm-up --niters 1 --grid 256 --nsteps 1000
ifeq ($(VALIDATION),yes)
CPPFLAGS += -DHECBENCH_VALIDATION
endif
LAUNCHER ?=
all: main kernel.hip
main: main.cpp constants.cpp grid.cpp pml.cpp data_setup.cpp minimig.cpp constants.h grid.h pml.h data_setup.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ main.cpp constants.cpp grid.cpp pml.cpp data_setup.cpp minimig.cpp $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp constants.h grid.h Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ kernel.hip.cpp
kernel.ptx: kernel.cu
	nvcc -std=c++17 -ptx -o $@ $<
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
