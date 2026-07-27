CC = g++
OPTIMIZE = yes
LAUNCHER =

program = main
CFLAGS := -Wall -fopenmp
LDFLAGS = -lm -fopenmp

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

$(program): main.c
	$(CC) $(CFLAGS) main.c -o $(program) $(LDFLAGS)

clean:
	rm -rf $(program)

run: $(program)
	$(LAUNCHER) ./$(program)
