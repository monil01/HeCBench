IRIS_ROOT ?= /tmp/iris
IRIS_LIBDIR ?= $(IRIS_ROOT)/build/src/runtime
CXX = g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Wall
HIPFLAGS ?= -O3 -ffast-math -std=c++17
RUN_ARGS ?= -f ../bitcracker-cuda/hash_pass/img_win8_user_hash.txt -d ../bitcracker-cuda/hash_pass/user_passwords_60000.txt -b 60000
.PHONY: all run clean
all: main kernel.hip
main: main.cpp attack.cpp w_blocks.cpp utils.cpp bitcracker.h
	$(CXX) $(CXXFLAGS) -I$(IRIS_ROOT)/include main.cpp attack.cpp w_blocks.cpp utils.cpp -o $@ -L$(IRIS_LIBDIR) -Wl,-rpath,$(IRIS_LIBDIR) -liris
kernel.hip: kernel.hip.cpp bitcracker.h sha256.h aes.h
	$(HIPCC) $(HIPFLAGS) --genco kernel.hip.cpp -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
