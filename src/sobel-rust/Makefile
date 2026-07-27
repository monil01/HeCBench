CARGO ?= cargo
LAUNCHER ?=

main:
	$(CARGO) build --release

clean:
	$(CARGO) clean

run: main
	$(LAUNCHER) ./target/release/sobel-rust 512 512 100
