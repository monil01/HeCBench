.PHONY: verify
verify:
	cd ../.. && python3 src/braycurtis-julia/prepare_cold_config.py --output /tmp/braycurtis-cold-config.json
	cd ../.. && python3 tools/verify_portable_julia.py braycurtis --config /tmp/braycurtis-cold-config.json
