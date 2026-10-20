.PHONY: config host
config:
	python3 make_config.py --output study_config_local.json
host:
	flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 python3 build_library.py study_host.so
