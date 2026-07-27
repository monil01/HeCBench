CC = nvc++
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
SM        = cc120
LAUNCHER  =

program = main
source = hotspot.cpp
obj = $(source:.cpp=.o)

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall
LDFLAGS =

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS += -mp=gpu -gpu=$(SM) -Minfo=mp
else
  CFLAGS += -mp=multicore
endif

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp hotspot.h reference.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) output_512.out

run: $(program)
	$(LAUNCHER) ./$(program) 512 2 200 ../data/hotspot/temp_512 ../data/hotspot/power_512 output_512.out
