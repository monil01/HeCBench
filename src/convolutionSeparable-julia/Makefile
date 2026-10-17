JULIA ?= julia
PROJECT ?= --project=../_julia_ka_env
LAUNCHER ?=
ARGS ?= 1024 1024 10

main:
	@true

clean:
	rm -rf __pycache__

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl $(ARGS)
