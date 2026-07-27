CARGO ?= cargo
LAUNCHER ?=

main:
	$(CARGO) build --release

clean:
	$(CARGO) clean

run: main
	$(LAUNCHER) ./target/release/black-scholes-rust 100
