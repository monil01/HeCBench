JULIA ?= julia
PROJECT ?= --project=../_julia_ka_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl ../boxfilter-sycl/data/lenaRGB.ppm 10000
