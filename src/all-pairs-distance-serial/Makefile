CC = gcc
program = main
CFLAGS := -Wall -O3

$(program): main.c
	$(CC) $(CFLAGS) main.c -o $(program)

clean:
	rm -rf $(program)

run: $(program)
	./$(program) 5
