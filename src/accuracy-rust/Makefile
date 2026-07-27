CARGO ?= cargo
LAUNCHER ?=

main:
	$(CARGO) build --release

clean:
	$(CARGO) clean

run: main
	$(LAUNCHER) $(CARGO) run --release -- 8192 10000 10 100
