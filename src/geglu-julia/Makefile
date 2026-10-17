# Named verified configuration; REPEAT controls the internal repeats, not the archived historical problem-size CLI.
JULIA ?= julia
PROJECT ?= --project=../_julia_math_ka_env
LAUNCHER ?=
REPEAT ?= 100
.PHONY: main run clean
main:
	@true
run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl $(REPEAT)
clean:
	rm -f *_trial*.bin
