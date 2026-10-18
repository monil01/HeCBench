CPU_PYTHON ?= python3
fixtures:
	OPENBLAS_NUM_THREADS=1 $(CPU_PYTHON) generate_fixtures.py
verify:
	OPENBLAS_NUM_THREADS=1 $(CPU_PYTHON) ../../tools/verify_portable_julia.py --config study_config.json
run: verify
cold-config:
	$(CPU_PYTHON) prepare_cold_config.py
