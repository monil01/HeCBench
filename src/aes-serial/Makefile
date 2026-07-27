CC = gcc
OPTIMIZE = yes
LAUNCHER =

program = main
CFLAGS := -Wall
LDFLAGS =

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

$(program): main.c
	$(CC) $(CFLAGS) main.c -o $(program) $(LDFLAGS)

clean:
	rm -rf $(program)

run: $(program)
	$(LAUNCHER) ./$(program) 100 0 ../urng-sycl/URNG_Input.bmp
