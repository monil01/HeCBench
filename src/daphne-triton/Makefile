PYTHON ?= /noback/imo/miniconda3/bin/python3
LAUNCHER ?=

$(shell mkdir -p .stamps)
.stamps/build:
	$(PYTHON) -c "import triton, torch; assert torch.cuda.is_available()"
	@touch $@

main: .stamps/build

clean:
	rm -rf .stamps __pycache__

run: main
	$(LAUNCHER) $(PYTHON) main.py -p 1
