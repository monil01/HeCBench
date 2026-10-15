IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
HIPCC ?= hipcc
RUN_ARGS ?=
CXXFLAGS ?= -O3 -std=c++17
all: main kernel.hip
main: main.cpp Makefile
	$(HIPCC) $(CXXFLAGS) -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include main.cpp -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) -O3 --genco $< -o $@
run: all
	@if [ -n "$(RUN_ARGS)" ]; then IRIS_ARCHS=HIP ./main $(RUN_ARGS); else IRIS_ARCHS=HIP ./main 4096 4096 4096 100 && IRIS_ARCHS=HIP ./main 8192 8192 8192 100; fi
clean:
	rm -f main kernel.hip *.o
