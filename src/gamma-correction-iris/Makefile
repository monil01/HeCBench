IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
OPTIMIZE ?= yes
LAUNCHER ?=
RUN_ARGS ?= 7680 4320 256 100

CXXFLAGS := -std=c++17 -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include -Wall
HIPFLAGS :=
LDFLAGS := -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread

ifeq ($(OPTIMIZE),yes)
  CXXFLAGS += -O3
  HIPFLAGS += -O3
endif

all: main kernel.hip

main: main.cpp Makefile utils.hpp utils/Img.hpp utils/ImgAlgorithm.hpp utils/ImgFormat.hpp utils/ImgPixel.hpp utils/Other.hpp
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(LDFLAGS)

kernel.hip: kernel.hip.cpp Makefile utils/ImgPixel.hpp
	$(HIPCC) $(HIPFLAGS) --genco -o $@ kernel.hip.cpp

kernel.ptx: kernel.cu Makefile utils/ImgPixel.hpp
	nvcc -ptx -o $@ kernel.cu

run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)

clean:
	rm -f main kernel.hip kernel.ptx *.o
