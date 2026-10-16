JULIA ?= julia
PROJECT ?= --project=$(abspath ../_julia_env)
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 10000000 100000
