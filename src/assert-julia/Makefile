PYTHON ?= python3
verify:
	$(PYTHON) ../../tools/verify_portable_julia.py --config study_config.json
run: verify
audit-saved:
	$(PYTHON) audit_execution.py
