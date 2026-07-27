CARGO ?= cargo
LAUNCHER ?=

main:
	$(CARGO) build --release --quiet

clean:
	$(CARGO) clean --quiet

run: main
	$(LAUNCHER) $(CARGO) run --release --quiet -- 4194304 100
