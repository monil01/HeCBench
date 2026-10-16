JULIA ?= julia
PROJECT ?= --project=../_julia_ka_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 2048 2048 96 100
