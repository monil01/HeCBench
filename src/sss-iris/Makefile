IRIS_ROOT ?= /tmp/iris
IRIS_LIBDIR ?= $(IRIS_ROOT)/build/src/runtime
GSL ?= /path/to/GSL
CXX = hipcc
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
HIPFLAGS ?= -O3 -std=c++17
RUN_ARGS ?= f9_n150_p50
DATA_DIR ?= ../sss-cuda/DATA
.PHONY: all run clean
all: main kernel.hip
main: main.cpp dmem.hpp DPmixGGM.cpp DPmixGGM_Lists.cpp DPmixGGM_SSSmoves.cpp graph.cpp graph.h gwish.cpp utilities.cpp
	$(CXX) $(CXXFLAGS) -I$(IRIS_ROOT)/include -I$(GSL)/include main.cpp -o $@ -L$(IRIS_LIBDIR) -Wl,-rpath,$(IRIS_LIBDIR) -liris -L$(GSL)/lib -lgsl -lgslcblas -lm
kernel.hip: kernel.hip.cpp
	$(HIPCC) $(HIPFLAGS) --genco $< -o $@
run: all
	mkdir -p RES
	test -e DATA || test -L DATA || ln -s $(DATA_DIR) DATA
	GSL_RNG_SEED=123 IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
