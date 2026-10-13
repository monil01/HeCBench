IRIS_ROOT ?= /tmp/iris
IRIS_LIBDIR ?= $(IRIS_ROOT)/build/src/runtime
ifeq ($(origin CXX),default)
CXX = hipcc
endif
HIPCC ?= hipcc
NVCC ?= nvcc
CXXFLAGS ?= -O3 -std=c++17 -Wall
CPPFLAGS += -I$(IRIS_ROOT)/include
LDFLAGS += -L$(IRIS_LIBDIR) -Wl,-rpath,$(IRIS_LIBDIR)
LDLIBS += -liris
HIPFLAGS ?=
RUN_ARGS ?=

.PHONY: all clean run cuda
all: main kernel.hip
main: main.cpp config.h reference.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.cpp -o $@ $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp
	$(HIPCC) $(HIPFLAGS) -O3 -std=c++17 --genco $< -o $@
cuda: kernel.cu
	$(NVCC) -O3 -std=c++17 -ptx $< -o kernel.ptx
run: all
ifneq ($(strip $(RUN_ARGS)),)
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
else
	IRIS_ARCHS=HIP ./main --op bgmv_shrink --tokens 1 --hidden 4096 --rank 16 --loras 4 --repeat 200
	IRIS_ARCHS=HIP ./main --op bgmv_shrink --vectorize --tokens 1 --hidden 4096 --rank 16 --loras 4 --repeat 200
	IRIS_ARCHS=HIP ./main --op bgmv_shrink --tokens 512 --hidden 4096 --rank 16 --loras 4 --repeat 200
	IRIS_ARCHS=HIP ./main --op bgmv_shrink --vectorize --tokens 512 --hidden 4096 --rank 16 --loras 4 --repeat 200
	IRIS_ARCHS=HIP ./main --op bgmv_expand --tokens 1 --hidden 4096 --rank 16 --loras 4 --repeat 200
	IRIS_ARCHS=HIP ./main --op bgmv_expand --vectorize --tokens 1 --hidden 4096 --rank 16 --loras 4 --repeat 200
	IRIS_ARCHS=HIP ./main --op bgmv_expand --tokens 512 --hidden 4096 --rank 16 --loras 4 --repeat 200
	IRIS_ARCHS=HIP ./main --op bgmv_expand --vectorize --tokens 512 --hidden 4096 --rank 16 --loras 4 --repeat 200
endif
clean:
	rm -f main kernel.hip kernel.ptx *.o
