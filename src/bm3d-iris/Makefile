IRIS_ROOT ?= /tmp/iris
IRIS_LIBDIR ?= $(IRIS_ROOT)/build/src/runtime
CXX = hipcc
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
HIPFLAGS ?= -O3 -std=c++17
EXTRA_CFLAGS ?=
CIMG_DIR ?= ../bm3d-cuda
RUN_ARGS ?= ../bm3d-cuda/noised-bufferfly-20.png denoised.png 20 color ../bm3d-cuda/bufferfly.png
.PHONY: all run clean
all: main kernel.hip
main: main.cpp launchers.cpp bm3d.hpp dmem.hpp indices.hpp params.hpp
	$(CXX) $(CXXFLAGS) $(EXTRA_CFLAGS) -I$(CIMG_DIR) -I$(IRIS_ROOT)/include main.cpp launchers.cpp -o $@ -L$(IRIS_LIBDIR) -Wl,-rpath,$(IRIS_LIBDIR) -liris -lpthread
kernel.hip: kernel.hip.cpp indices.hpp params.hpp
	$(HIPCC) $(HIPFLAGS) --genco kernel.hip.cpp -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o denoised.png pixels.bin
