.PHONY: verify
verify:
	cd ../.. && python3 src/cc-julia/prepare_cold_config.py --output /tmp/cc-cold-config.json
	cd ../.. && python3 tools/verify_portable_julia.py cc --config /tmp/cc-cold-config.json
