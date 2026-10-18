PYTHON ?= python3
verify:
	$(PYTHON) ../../tools/verify_portable_julia.py --config study_config.json
run: verify
reproduce-cpu:
	$(PYTHON) reproduce_assets.py
cold-config:
	$(PYTHON) prepare_cold_config.py
