JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl --op bgmv_shrink --tokens 1 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl --op bgmv_shrink --vectorize --tokens 1 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl --op bgmv_shrink --tokens 512 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl --op bgmv_shrink --vectorize --tokens 512 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl --op bgmv_expand --tokens 1 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl --op bgmv_expand --vectorize --tokens 1 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl --op bgmv_expand --tokens 512 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl --op bgmv_expand --vectorize --tokens 512 --hidden 4096 --rank 16 --loras 4 --repeat 200
