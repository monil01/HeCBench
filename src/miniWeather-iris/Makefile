IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX = g++
HIPCC ?= hipcc
MPI_INCLUDE ?= /usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIB_DIR ?= /usr/lib/x86_64-linux-gnu/openmpi/lib
NX ?= 400
NZ ?= 200
SIM_TIME ?= 600
DATA_SPEC ?= DATA_SPEC_THERMAL
CONFIG = -DNX=$(NX) -DNZ=$(NZ) -DSIM_TIME=$(SIM_TIME) -DDATA_SPEC=$(DATA_SPEC) -DOMPI_SKIP_MPICXX=1
RUN_ARGS ?=
LAUNCHER ?=
all: main kernel.hip
main: main.cpp $(wildcard *.h)
	$(CXX) -O3 -std=c++17 $(CONFIG) -I$(MPI_INCLUDE) -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include main.cpp -o $@ -L$(MPI_LIB_DIR) -Wl,-rpath,$(MPI_LIB_DIR) -lmpi -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -lpthread -ldl
kernel.hip: kernel.hip.cpp constants.h kernels.h
	$(HIPCC) -O3 --genco kernel.hip.cpp -o $@
kernel.ptx: kernel.cu constants.h kernels.h
	nvcc -O3 -ptx kernel.cu -o $@
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
