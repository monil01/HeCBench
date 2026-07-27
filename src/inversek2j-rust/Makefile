CARGO ?= cargo
LAUNCHER ?=

main:
	$(CARGO) build --release

clean:
	$(CARGO) clean

run: main
	$(LAUNCHER) ./target/release/inversek2j-rust ../inversek2j-sycl/coord_in.txt 100000
