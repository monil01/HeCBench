PYTHON ?= python3
fixtures:
	OPENBLAS_NUM_THREADS=1 $(PYTHON) generate_fixtures.py
verify:
	OPENBLAS_NUM_THREADS=1 $(PYTHON) ../../tools/verify_portable_julia.py --config study_config.json
run: verify
