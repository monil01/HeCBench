MOJO ?= mojo
LAUNCHER ?=

main:
	@true

clean:
	@true

# aop-cuda runs 100 Longstaff-Schwartz iterations; this Mojo port is a
# Monte-Carlo European surrogate — a handful of runs is enough.
run: main
	$(LAUNCHER) $(MOJO) main.mojo -runs 3
