CARGO ?= cargo
LAUNCHER ?=

main:
	$(CARGO) build --release

clean:
	$(CARGO) clean

run: main
	$(LAUNCHER) ./target/release/softmax-rust 100000 784 0 100
