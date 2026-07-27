CARGO ?= cargo
LAUNCHER ?=

main:
	$(CARGO) build --release

clean:
	$(CARGO) clean

run: main
	$(LAUNCHER) $(CARGO) run --release -- ../affine-cuda/data/CT-MONO2-16-brain.raw result.raw 1000000
