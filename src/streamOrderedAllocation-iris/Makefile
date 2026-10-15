IRIS_ROOT ?= /tmp/iris
IRIS_LIBDIR ?= $(IRIS_ROOT)/build/src/runtime
CXX = hipcc
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Wall
RUN_ARGS ?=
all: main kernel.hip
main: main.cpp Makefile
	$(CXX) $(CXXFLAGS) -I$(IRIS_ROOT)/include $< -L$(IRIS_LIBDIR) -Wl,-rpath,$(IRIS_LIBDIR) -liris -o $@
kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) -O3 --genco $< -o $@
runtime_test: runtime_test.cpp Makefile
	$(CXX) $(CXXFLAGS) -I$(IRIS_ROOT)/include $< -L$(IRIS_LIBDIR) -Wl,-rpath,$(IRIS_LIBDIR) -liris -o $@
run: all
	IRIS_ARCHS=HIP IRIS_ASYNC=1 ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip runtime_test *.o
.PHONY: all run clean
