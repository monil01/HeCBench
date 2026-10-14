IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
NVCC ?= nvcc
CXXFLAGS ?= -O3 -std=c++17
POINTS ?= 97178
DATAPARS ?= -d ./data/small/Datapnts.1 -p $(POINTS)
RANDPARS ?= -r ./data/small/Randompnts -n 100 -q $(POINTS)
ANALPARS ?= -b 5 -l 1 -u 10000 -a -j 10
OUTFILE ?= ./small_results_test.dat
RUN_ARGS ?= $(DATAPARS) $(RANDPARS) $(ANALPARS) -o $(OUTFILE)
all: main kernel.hip
main: main.cpp compute.cpp args.cpp args.h kernel.h backend.h model_io.h dump.h
	$(CXX) $(CXXFLAGS) -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include main.cpp compute.cpp args.cpp -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
kernel.hip: kernel.hip.cpp device_kernels.h kernel.h
	$(HIPCC) -O3 --genco -o $@ $<
kernel.ptx: kernel.cu device_kernels.h kernel.h
	$(NVCC) -ptx -o $@ $<
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
