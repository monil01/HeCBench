DATA_ROOT ?= ../../.porting-state/portable_julia/external_assets/egs/data
.PHONY: data config
data:
	python3 regenerate.py --data-root $(DATA_ROOT)
config:
	python3 make_config.py --data-root $(DATA_ROOT) --output /tmp/egs_fourway.json
