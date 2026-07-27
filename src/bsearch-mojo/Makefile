MOJO ?= mojo
LAUNCHER ?=

main:
	@true

clean:
	@true

# 8388608 = 8M keys (matches CUDA benchmark).
run: main
	$(LAUNCHER) $(MOJO) main.mojo 8388608 100
