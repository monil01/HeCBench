PYTHON ?= /noback/imo/miniconda3/bin/python3
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__ output_512.out

run: main
	$(LAUNCHER) $(PYTHON) main.py 512 2 200 ../data/hotspot/temp_512 ../data/hotspot/power_512 output_512.out
