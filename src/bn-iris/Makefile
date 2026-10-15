IRIS_ROOT ?= /tmp/iris
CXX = g++
HIPCC ?= hipcc
HIPFLAGS ?= -O1
CXXFLAGS ?= -O3 -std=c++17 -fopenmp
EXTRA_CFLAGS ?=
CPPFLAGS += $(EXTRA_CFLAGS) -I$(IRIS_ROOT)/include -I$(IRIS_ROOT)/build/include
LDFLAGS += -L$(IRIS_ROOT)/build/src/runtime -Wl,-rpath,$(IRIS_ROOT)/build/src/runtime -liris
RUN_ARGS ?= result 10
all: main kernel.hip
main: main.cpp reference.h data45.h data45_original.h dmem.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) main.cpp -o $@ $(LDFLAGS)
kernel.hip: kernel.hip.cpp data45.h data45_original.h
	$(HIPCC) $(HIPFLAGS) $(EXTRA_CFLAGS) --genco $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o result
.PHONY: all run clean
