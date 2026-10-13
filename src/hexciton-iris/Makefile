IRIS_ROOT ?= /tmp/iris
IRIS_LIBDIR ?= $(IRIS_ROOT)/build/src/runtime
CXX = hipcc
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
HIPFLAGS ?= -O3 -std=c++17
EXTRA_CFLAGS ?=
RUN_ARGS ?=
.PHONY: all run clean
all: main kernel.hip
main: main.cpp utils.cpp reference.cpp utils.hpp
	$(CXX) $(CXXFLAGS) $(EXTRA_CFLAGS) -I$(IRIS_ROOT)/include main.cpp utils.cpp reference.cpp -o $@ -L$(IRIS_LIBDIR) -Wl,-rpath,$(IRIS_LIBDIR) -liris
kernel.hip: kernel.hip.cpp utils.hpp
	$(HIPCC) $(HIPFLAGS) $(EXTRA_CFLAGS) --genco kernel.hip.cpp -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
