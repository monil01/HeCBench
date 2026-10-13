IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX = g++
HIPCC ?= hipcc
REFDATA_DIR ?= ../testSNAP-omp
REFDATA_TWOJ ?= 14
RUN_ARGS ?= --nsteps 100
LAUNCHER ?=
all: main kernel.hip
main: main.cpp snap.h utils_host.h launch.h backend.h $(REFDATA_DIR)/refdata_2J$(REFDATA_TWOJ)_W.h
	$(CXX) -O3 -std=c++17 -DREFDATA_TWOJ=$(REFDATA_TWOJ) -I$(REFDATA_DIR) -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include main.cpp -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -lpthread -ldl
kernel.hip: kernel.hip.cpp snap.h utils_device.h kernels.h
	$(HIPCC) -O3 --genco kernel.hip.cpp -o $@
kernel.ptx: kernel.cu snap.h utils_device.h kernels.h
	nvcc -O3 -ptx kernel.cu -o $@
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
