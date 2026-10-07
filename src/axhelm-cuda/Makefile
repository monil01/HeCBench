#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

obj = main.o meshBasis.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH) \
          -Ddfloat=float -Ddlong=int 

# Linker Flags
LDFLAGS = -L/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/lib -llapack -lblas

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $+ -o $@ $(LDFLAGS)

main.o : main.cu axhelmKernel.cpp axhelmReference.cpp
	$(CC) $(CFLAGS) -c $< -o $@
	
meshBasis.o : meshBasis.cpp meshBasis.hpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

# run one- and three-dimensional kernels
run: $(program)
	$(LAUNCHER) ./$(program) 1 8000 100
	$(LAUNCHER) ./$(program) 3 8000 100
