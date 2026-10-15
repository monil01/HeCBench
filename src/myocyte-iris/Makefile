IRIS_ROOT ?= /tmp/iris
IRIS_LIBDIR ?= $(IRIS_ROOT)/build/src/runtime
CXX = hipcc
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
HIPFLAGS ?= -O3 -std=c++17
RUN_ARGS ?= 100
.PHONY: all run clean
all: main kernel.hip
main: main.cpp work.cpp master.cpp solver.cpp embedded_fehlberg_7_8.cpp kernel_fin.cu file.c define.h
	$(CXX) $(CXXFLAGS) -I$(IRIS_ROOT)/include main.cpp -o $@ -L$(IRIS_LIBDIR) -Wl,-rpath,$(IRIS_LIBDIR) -liris
kernel.hip: kernel.hip.cpp kernel_ecc.cu kernel_cam.cu define.h
	$(HIPCC) $(HIPFLAGS) --genco kernel.hip.cpp -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o output.txt
