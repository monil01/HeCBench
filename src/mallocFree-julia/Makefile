PYTHON ?= python3
runtime:
	$(PYTHON) build_runtime.py
run: runtime
	cd ../.. && $(PYTHON) tools/verify_portable_julia.py --config src/mallocFree-julia/study_config.json
