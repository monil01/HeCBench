JULIA ?= julia
PROJECT ?= --project=../_julia_ka_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 224 224 100
