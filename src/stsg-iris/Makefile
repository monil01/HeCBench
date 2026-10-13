IRIS_ROOT ?= /tmp/iris
GDAL_INCLUDE ?= /usr/include
GDAL_LIB ?= -lgdal
CXX = g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_ROOT)/build/include -I$(GDAL_INCLUDE)
LDFLAGS += -L$(IRIS_ROOT)/build/src/runtime -Wl,-rpath,$(IRIS_ROOT)/build/src/runtime -liris $(GDAL_LIB) -ldl
RUN_ARGS ?= example.txt
all: main kernel.hip
main: main.cpp dmem.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) main.cpp -o $@ $(LDFLAGS)
kernel.hip: kernel.hip.cpp
	$(HIPCC) -O3 --genco $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
