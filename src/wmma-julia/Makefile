JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	#$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 0 16 16 16 100 1
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 0 16 16 64 100 1
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 0 32 32 64 100 1
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 0 256 256 256 100 1
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 1 256 256 256 100 1
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 0 512 512 512 100 1
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 1 512 512 512 100 1
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 0 1024 1024 1024 100 0
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 1 1024 1024 1024 100 0
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 0 4096 4096 4096 100 0
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 1 4096 4096 4096 100 0
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 0 8192 8192 8192 100 0
	@$(LAUNCHER) $(JULIA) $(PROJECT) main.jl 1 8192 8192 8192 100 0
