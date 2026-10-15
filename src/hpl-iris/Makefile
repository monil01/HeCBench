IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
HIPCC ?= hipcc
MPICC ?= /usr/bin/mpicc
LAUNCHER ?= /usr/bin/mpirun -n 1
HPL_ROOT ?= ../hpl-cuda/src/hpl-2.3
HPL_NATIVE_ROOT ?= ../hpl-hip/src/hpl-2.3
HPL_INPUT ?= ../hpl-cuda/datafiles/HPL_small_gpu.dat
BLAS_LIBS ?= -lblas
MPI_CFLAGS ?= $(shell $(MPICC) --showme:compile)
MPI_LIBS ?= $(shell $(MPICC) --showme:link)
CFLAGS ?= -O3 -funroll-loops
CXXFLAGS ?= -O3 -std=c++17 -DOMPI_SKIP_MPICXX=1
HPL_DEFS = -DHPL_CALL_CBLAS -DAdd__ -DF77_INTEGER=int -DStringSunStyle
C_SOURCES := $(wildcard $(HPL_ROOT)/src/*/*.c $(HPL_ROOT)/testing/*/*.c)
NATIVE_C_SOURCES := $(wildcard $(HPL_NATIVE_ROOT)/src/*/*.c $(HPL_NATIVE_ROOT)/testing/*/*.c)
OBJECTS := $(patsubst $(HPL_ROOT)/%.c,.build/hpl/%.o,$(C_SOURCES))
NATIVE_OBJECTS := $(patsubst $(HPL_NATIVE_ROOT)/%.c,.build/native/%.o,$(NATIVE_C_SOURCES))
WRAP_LIBS = -Wl,--wrap=HPL_pdgesv -Wl,--wrap=hipblasDgemm -Wl,--wrap=hipblasDtrsm
IRIS_FLAGS = -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
IRIS_LIBS = -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
.PHONY: all native run native-run clean
all: main libhecbench_vendor.so
native: native_main
.build/hpl/%.o: $(HPL_ROOT)/%.c Makefile
	mkdir -p $(dir $@)
	$(MPICC) $(CFLAGS) $(HPL_DEFS) -I$(HPL_ROOT)/include $(if $(filter %/HPL_pddriver.c,$<),-Dmain=hpl_driver_main,) -c $< -o $@
.build/native/%.o: $(HPL_NATIVE_ROOT)/%.c Makefile
	mkdir -p $(dir $@)
	$(MPICC) $(CFLAGS) $(HPL_DEFS) -I$(HPL_NATIVE_ROOT)/include $(if $(filter %/HPL_pddriver.c,$<),-Dmain=hpl_driver_main,) -c $< -o $@
.build/observe.o: observe.c Makefile
	mkdir -p .build
	$(MPICC) $(CFLAGS) $(HPL_DEFS) -I$(HPL_ROOT)/include -c $< -o $@
.build/vendor_counts.o: vendor_counts.cpp Makefile
	mkdir -p .build
	$(HIPCC) $(CXXFLAGS) -c $< -o $@
.build/main.o: main.cpp ../../tools/iris_vendor_bridge.hpp Makefile
	mkdir -p .build
	$(HIPCC) $(CXXFLAGS) $(MPI_CFLAGS) $(IRIS_FLAGS) -c $< -o $@
.build/native_entry.o: native_entry.cpp Makefile
	mkdir -p .build
	$(HIPCC) $(CXXFLAGS) -c $< -o $@
.build/native_gpu.o: $(HPL_NATIVE_ROOT)/src/cuda/cuda_dgemm.cpp Makefile
	mkdir -p .build
	$(HIPCC) $(CXXFLAGS) -c $< -o $@
main: $(OBJECTS) .build/main.o .build/observe.o .build/vendor_counts.o
	$(HIPCC) $(CXXFLAGS) -o $@ $^ $(WRAP_LIBS) -Wl,--wrap=MPI_Init -Wl,--wrap=MPI_Finalize $(MPI_LIBS) $(IRIS_LIBS) -lhipblas $(BLAS_LIBS) -lm
native_main: $(NATIVE_OBJECTS) .build/native_entry.o .build/native_gpu.o .build/observe.o .build/vendor_counts.o
	$(HIPCC) $(CXXFLAGS) -o $@ $^ $(WRAP_LIBS) $(MPI_LIBS) -lhipblas $(BLAS_LIBS) -lm
libhecbench_vendor.so: kernel.hip.cpp ../../tools/iris_vendor_bridge.cpp ../../tools/iris_vendor_bridge.hpp
	$(HIPCC) -O2 -std=c++17 -fPIC -shared $< -o $@
run: all
	mkdir -p .run
	cp $(HPL_INPUT) .run/HPL.dat
	cd .run && IRIS_ARCHS=HIP IRIS_KERNEL_HOST2HIP=$(abspath libhecbench_vendor.so) $(LAUNCHER) ../main
native-run: native
	mkdir -p .run-native
	cp $(HPL_INPUT) .run-native/HPL.dat
	cd .run-native && $(LAUNCHER) ../native_main
clean:
	rm -f main native_main libhecbench_vendor.so $(OBJECTS) $(NATIVE_OBJECTS) .build/main.o .build/observe.o .build/vendor_counts.o .build/native_entry.o .build/native_gpu.o
