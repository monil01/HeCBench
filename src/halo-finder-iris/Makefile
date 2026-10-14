IRIS_ROOT ?= /tmp/iris
HIP_ROOT ?= /opt/rocm
CXX ?= g++
CC ?= gcc
HIPCC ?= $(HIP_ROOT)/bin/hipcc
MPI_INCLUDE ?= /usr/lib/x86_64-linux-gnu/openmpi/include
RUN_ARGS ?= 0.5 0.1 10000 0.1 10 N 12 rcb
CPPFLAGS += -DRCB_UNTHREADED_BUILD -DUSE_SERIAL_COSMO -DOMPI_SKIP_MPICXX -DHECBENCH_IRIS -DID_64 -DPOSVEL_32 -DGRID_32 -DLONG_INTEGER -DPENCIL=1 -D__HIP_PLATFORM_AMD__ -I. -I$(MPI_INCLUDE) -I$(HIP_ROOT)/include -I$(IRIS_ROOT)/include
CXXFLAGS ?= -O3 -std=c++17 -pthread
CFLAGS ?= -O3 -std=c99 -pthread
HOST_CPP = main.cpp ForceLaw.cxx BHForceTree.cxx RCOForceTree.cxx RCBForceTree.cxx ParticleDistribute.cxx ParticleExchange.cxx InitialExchange.cxx Message.cxx Partition.cxx Timings.cxx Timer.cxx
HOST_C = bigchunk.c BGQStep16.c dims-local.c
OBJECTS = $(HOST_CPP:.cxx=.o)
OBJECTS := $(OBJECTS:.cpp=.o) $(HOST_C:.c=.o)
all: main kernel.hip
main: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(IRIS_ROOT)/build/src/runtime -Wl,-rpath,$(IRIS_ROOT)/build/src/runtime -liris -lm -lrt
%.o: %.cxx Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
%.o: %.cpp Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
%.o: %.c Makefile
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
kernel.hip: kernel.hip.cpp BasicDefinition.h
	$(HIPCC) -O3 -DID_64 -DPOSVEL_32 -DGRID_32 --genco $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip $(OBJECTS)
