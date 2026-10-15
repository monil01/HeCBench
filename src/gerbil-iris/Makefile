IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
ROCM_PATH ?= /opt/rocm
CXX ?= g++
HIPCC ?= hipcc
BOOST_LIBDIR ?= $(CURDIR)/.deps/boost/usr/lib/x86_64-linux-gnu
BOOST_VERSION ?= 1.83.0
BOOST_LIBS ?= $(foreach lib,system thread regex filesystem atomic,-l:libboost_$(lib).so.$(BOOST_VERSION))
RUN_ARGS ?= -i -g ERR164407.fastq.gz tmp out
CPP_SOURCES := $(filter-out native/src/gerbil/toFasta.cpp,$(wildcard native/src/gerbil/*.cpp))
HIP_SOURCES := $(wildcard native/src/cuda_ds/*.cu)
CPP_OBJECTS := $(CPP_SOURCES:.cpp=.o)
HIP_OBJECTS := $(HIP_SOURCES:.cu=.o)
HEADERS := $(shell find native/include -type f -name '*.h')
COMMON := -O3 -std=c++17 -fPIC -DGERBIL_USE_GPU -D__HIP_PLATFORM_AMD__ -I. -I$(ROCM_PATH)/include -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
CXXFLAGS += $(COMMON) -fpermissive -DBOOST_ALL_NO_LIB -DBOOST_ATOMIC_DYN_LINK -DBOOST_FILESYSTEM_DYN_LINK -DBOOST_REGEX_DYN_LINK -DBOOST_SYSTEM_DYN_LINK -DBOOST_THREAD_DYN_LINK
HIPFLAGS += $(COMMON)
LIBS := -L$(BOOST_LIBDIR) -Wl,-rpath,$(BOOST_LIBDIR) $(BOOST_LIBS) -lbz2 -lz -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64 -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
all: main libhecbench_vendor.so
native/src/gerbil/%.o: native/src/gerbil/%.cpp $(HEADERS) dmem_adapter.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
native/src/cuda_ds/%.o: native/src/cuda_ds/%.cu $(HEADERS) dmem_adapter.hpp
	$(HIPCC) $(HIPFLAGS) -c $< -o $@
dmem_adapter.o: dmem_adapter.cpp dmem_adapter.hpp
	$(HIPCC) $(HIPFLAGS) -c $< -o $@
libgerbil.so: $(CPP_OBJECTS) $(HIP_OBJECTS) dmem_adapter.o
	$(HIPCC) -shared $^ -o $@ $(LIBS)
main: main.cpp libgerbil.so
	$(CXX) $(CXXFLAGS) $< -L. -Wl,-rpath,'$$ORIGIN' -lgerbil -o $@ $(LIBS)
libhecbench_vendor.so: kernel.hip.cpp ../../tools/iris_vendor_bridge.cpp
	$(CXX) -std=c++17 -fPIC -shared $< -o $@
run: all
	LD_LIBRARY_PATH=$(BOOST_LIBDIR):$(ROCM_PATH)/lib HIP_VISIBLE_DEVICES=0 IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main *.o *.so $(CPP_OBJECTS) $(HIP_OBJECTS)
.PHONY: all run clean
