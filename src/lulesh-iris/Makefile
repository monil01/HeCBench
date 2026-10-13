IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX = g++
HIPCC ?= hipcc
VERIFY ?= no
RUN_ARGS ?= -i 100 -s 128 -r 11 -b 1 -c 1
LAUNCHER ?=
ifeq ($(VERIFY),yes)
VERIFY_FLAGS = -DVERIFY
endif
all: main kernel.hip
main: main.cpp lulesh-init.cpp lulesh-util.cpp lulesh-viz.cpp lulesh.h geometry.h launch.h backend.h
	$(CXX) -O3 -std=c++17 $(VERIFY_FLAGS) -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include main.cpp lulesh-init.cpp lulesh-util.cpp lulesh-viz.cpp -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -lpthread -ldl
kernel.hip: kernel.hip.cpp kernels.h geometry.h lulesh.h
	$(HIPCC) -O3 --genco kernel.hip.cpp -o $@
kernel.ptx: kernel.cu kernels.h geometry.h lulesh.h
	nvcc -O3 -ptx kernel.cu -o $@
test: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main -i 1 -s 32 -r 11 -b 1 -c 1
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
