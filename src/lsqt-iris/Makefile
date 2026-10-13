IRIS_ROOT ?= /tmp/iris
IRIS_LIBDIR ?= $(IRIS_ROOT)/build/src/runtime
CXX = hipcc
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -ffast-math -DDEBUG
HIPFLAGS ?= -O3 -std=c++17 -ffast-math -DDEBUG
RUN_ARGS ?= ../lsqt-cuda/examples/input.txt
SOURCES = main.cpp lsqt.cpp sigma.cpp vector.cpp hamiltonian.cpp model.cpp model_general.cpp model_lattice.cpp anderson.cpp charge.cpp
.PHONY: all run clean
all: main kernel.hip
main: $(SOURCES) $(wildcard *.h) dmem.hpp
	$(CXX) $(CXXFLAGS) -I$(IRIS_ROOT)/include $(SOURCES) -o $@ -L$(IRIS_LIBDIR) -Wl,-rpath,$(IRIS_LIBDIR) -liris
kernel.hip: kernel.hip.cpp
	$(HIPCC) $(HIPFLAGS) --genco $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
