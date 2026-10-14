IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
ORIGINAL ?= ../miniDGS-cuda
PARMETIS_ROOT ?= .deps/ParMetis-3.1
MPICC ?= /usr/bin/mpicc
MPICXX ?= /usr/bin/mpicxx
MPIEXEC ?= /usr/bin/mpiexec
HIPCC ?= hipcc
N ?= 6
NP ?= 2
RUN_ARGS ?= $(ORIGINAL)/Meshes/FS_K01022.neu
COMMON = -Dp_N=$(N) -DNDG3d -DCUDA -Iinclude -I$(ORIGINAL)/include -O3
CXXFLAGS += -std=c++17 $(COMMON) -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
CFLAGS += $(COMMON) -fcommon -Wno-incompatible-pointer-types
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread -lm
SUPPORT = Mesh3d Utils LoadBalance3d FacePair3d ParallelPairs BuildMaps3d StartUp3d
OBJECTS = $(addprefix obj/,$(addsuffix .o,$(SUPPORT))) obj/MaxwellsMPI3d.o obj/MaxwellsRun3d.o
all: main kernel.hip
main: main.cpp host.cpp include/fem.h $(OBJECTS) Makefile $(PARMETIS_ROOT)/libparmetis.a $(PARMETIS_ROOT)/libmetis.a
	$(MPICXX) $(CXXFLAGS) main.cpp host.cpp $(OBJECTS) -o $@ $(PARMETIS_ROOT)/libparmetis.a $(PARMETIS_ROOT)/libmetis.a $(LDFLAGS)
ifeq ($(PARMETIS_ROOT),.deps/ParMetis-3.1)
$(PARMETIS_ROOT)/libparmetis.a $(PARMETIS_ROOT)/libmetis.a &: $(ORIGINAL)/3rdParty/ParMetis-3.1/Makefile.in
	mkdir -p .deps
	test -d $(PARMETIS_ROOT) || cp -a $(ORIGINAL)/3rdParty/ParMetis-3.1 .deps/
	$(MAKE) -C $(PARMETIS_ROOT)/METISLib CC=$(MPICC) COPTIONS='-fcommon -Wno-incompatible-pointer-types'
	$(MAKE) -C $(PARMETIS_ROOT)/ParMETISLib CC=$(MPICC) COPTIONS='-fcommon -Wno-incompatible-pointer-types'
endif
obj/%.o: $(ORIGINAL)/src/%.c include/fem.h
	mkdir -p obj
	$(MPICC) $(CFLAGS) -c $< -o $@
obj/MaxwellsMPI3d.o: MaxwellsMPI3d.c include/fem.h
	mkdir -p obj
	$(MPICC) $(CFLAGS) -c $< -o $@
obj/MaxwellsRun3d.o: MaxwellsRun3d.c include/fem.h
	mkdir -p obj
	$(MPICC) $(CFLAGS) -c $< -o $@
kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) -O3 -std=c++17 -Dp_N=$(N) -DNDG3d -I$(ORIGINAL)/include --genco -o $@ $<
kernel.ptx: kernel.cu
	nvcc -O3 -Dp_N=$(N) -DNDG3d -I$(ORIGINAL)/include -ptx -o $@ $<
run: all
	IRIS_ARCHS=HIP $(MPIEXEC) -n $(NP) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx obj/*.o
.PHONY: all run clean
