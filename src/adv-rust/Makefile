CARGO ?= cargo
LAUNCHER ?=

main:
	$(CARGO) build --release --quiet

clean:
	$(CARGO) clean --quiet

run: main
	$(LAUNCHER) $(CARGO) run --release --quiet -- 7 15 100
