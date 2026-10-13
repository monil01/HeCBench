IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX = $(HIPCC)
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Wall
HIPFLAGS ?= -O3 -munsafe-fp-atomics
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
RUN_ARGS ?=
ifneq ($(VALIDATION_STEPS),)
CPPFLAGS += -DVALIDATION_STEPS=$(VALIDATION_STEPS)
endif
LAUNCHER ?=
all: main kernel.hip
main: main.cpp fileio.cpp geometry.cpp common.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ main.cpp fileio.cpp geometry.cpp $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp common.h Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ kernel.hip.cpp
kernel.ptx: kernel.cu
	nvcc -std=c++17 -ptx -o $@ $<
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
