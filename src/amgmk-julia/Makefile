.PHONY: verify
verify:
	cd ../.. && python3 src/amgmk-julia/prepare_cold_config.py --output /tmp/amgmk-cold-config.json
	cd ../.. && PYTHONPATH=/tmp/pasc-julia-plot-tools python3 tools/verify_portable_julia.py amgmk --config /tmp/amgmk-cold-config.json
