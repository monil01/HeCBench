JULIA ?= julia
PROJECT ?= --project=../_julia_ka_env
LAUNCHER ?=
REPEAT ?= 10
.PHONY:run
run:
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl $(REPEAT)
