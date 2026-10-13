IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX = g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -DPROFILE_TMP
RUN_ARGS ?= 10 20 256 65536 65536 1000 none output.txt 1
LAUNCHER ?=
all: main kernel.hip
main: main.cpp streamcluster.h streamcluster_cl.h point_struct.h
	$(CXX) $(CXXFLAGS) -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include main.cpp -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -lpthread -ldl
kernel.hip: kernel.hip.cpp point_struct.h
	$(HIPCC) -O3 -ffp-contract=off --genco kernel.hip.cpp -o $@
kernel.ptx: kernel.cu point_struct.h
	nvcc -O3 --fmad=false -ptx kernel.cu -o $@
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
