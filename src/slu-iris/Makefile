IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
NICSLU_ROOT ?= .deps/nicslu
CXX ?= g++
HIPCC ?= hipcc
VERIFY ?= yes
RUN_ARGS ?= -i $(NICSLU_ROOT)/demo/ASIC_100k.mtx
SOURCES = main.cpp numeric.cpp symbolic.cpp preprocess.cpp Timer.cpp
HEADERS = $(wildcard include/*.h) validation_export.h
CXXFLAGS += -O3 -std=c++17 -Wall -Iinclude -I$(NICSLU_ROOT)/include -I$(NICSLU_ROOT)/util -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
ifeq ($(VERIFY),yes)
CXXFLAGS += -DVERIFY
endif
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread -lm
LIBS = $(NICSLU_ROOT)/lib/nicslu.a $(NICSLU_ROOT)/util/nicslu_util.a
all: main kernel.hip
main: $(SOURCES) $(HEADERS) $(LIBS) Makefile
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $@ $(LIBS) $(LDFLAGS)
ifeq ($(NICSLU_ROOT),.deps/nicslu)
$(LIBS) &: ../slu-cuda/src/nicslu.tar.bz2 prepare_nicslu.py
	python3 prepare_nicslu.py ../slu-cuda/src/nicslu.tar.bz2 .deps
	$(MAKE) -C $(NICSLU_ROOT)/lib
	$(MAKE) -C $(NICSLU_ROOT)/util
endif
kernel.hip: kernel.hip.cpp include/type.h Makefile
	$(HIPCC) -O3 -std=c++17 -Iinclude --genco -o $@ $<
kernel.ptx: kernel.cu include/type.h
	nvcc -O3 -Iinclude -ptx -o $@ $<
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
test: all
	IRIS_ARCHS=HIP ./main -i $(NICSLU_ROOT)/test/add32.mtx
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run test clean
