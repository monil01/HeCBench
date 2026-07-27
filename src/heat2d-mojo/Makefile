MOJO ?= mojo
LAUNCHER ?=

main:
	@true

clean:
	@true

# The CUDA benchmark runs 4096x4096, 1000 iters; the Mojo port's CPU
# reference is single-threaded so we use a smaller default that still
# passes verification (kernel itself scales to any size).
run: main
	$(LAUNCHER) $(MOJO) main.mojo 512 512 100
