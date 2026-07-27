CC = gcc
program = main
CFLAGS := -Wall -O3
LDFLAGS = -lm

$(program): main.c
	$(CC) $(CFLAGS) main.c -o $(program) $(LDFLAGS)

clean:
	rm -rf $(program)

run: $(program)
	./$(program) 100000 10 5
