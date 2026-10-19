.PHONY: verify
verify:
	cd ../.. && python3 src/nn-julia/prepare_cold_config.py --output /tmp/nn-cold-config.json
	cd ../.. && python3 tools/verify_portable_julia.py --config /tmp/nn-cold-config.json nn
