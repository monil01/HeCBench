IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX = hipcc
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Wall
HIPFLAGS ?= -O3
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
RUN_ARGS ?=
LAUNCHER ?=
all: main kernel.hip
main: main.cpp reference.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.cpp -o $@ $(LDFLAGS)
kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ kernel.hip.cpp
run: all
ifneq ($(strip $(RUN_ARGS)),)
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
else
	IRIS_ARCHS=HIP $(LAUNCHER) ./main 0 16 16 64 100 1
	IRIS_ARCHS=HIP $(LAUNCHER) ./main 0 32 32 64 100 1
	IRIS_ARCHS=HIP $(LAUNCHER) ./main 0 256 256 256 100 1
	IRIS_ARCHS=HIP $(LAUNCHER) ./main 1 256 256 256 100 1
	IRIS_ARCHS=HIP $(LAUNCHER) ./main 0 512 512 512 100 1
	IRIS_ARCHS=HIP $(LAUNCHER) ./main 1 512 512 512 100 1
	IRIS_ARCHS=HIP $(LAUNCHER) ./main 0 1024 1024 1024 100 0
	IRIS_ARCHS=HIP $(LAUNCHER) ./main 1 1024 1024 1024 100 0
	IRIS_ARCHS=HIP $(LAUNCHER) ./main 0 4096 4096 4096 100 0
	IRIS_ARCHS=HIP $(LAUNCHER) ./main 1 4096 4096 4096 100 0
	IRIS_ARCHS=HIP $(LAUNCHER) ./main 0 8192 8192 8192 100 0
	IRIS_ARCHS=HIP $(LAUNCHER) ./main 1 8192 8192 8192 100 0
endif
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
