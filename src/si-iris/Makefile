IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
HIPCC ?= hipcc
CXX ?= g++
CPPFLAGS += -DFMT_HEADER_ONLY -Iinclude -I.deps/cxxopts/include -I.deps/fmt/include -I.deps/zipf -I.deps/progress/include/progresscpp
CXXFLAGS ?= -O3 -std=c++17
RUN_ARGS ?=
all: main generate_dataset libhecbench_vendor.so
.deps/ready: fetch_deps.py
	python3 fetch_deps.py
	touch $@
main: main.cpp include/io.hpp include/util.hpp include/host_timer.hpp .deps/ready Makefile
	$(HIPCC) $(CXXFLAGS) $(CPPFLAGS) -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include $< -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -lhipblas -ldl -lpthread
generate_dataset: generate_dataset.cpp .deps/ready include/io.hpp Makefile
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
libhecbench_vendor.so: kernel.hip.cpp ../../tools/iris_vendor_bridge.cpp ../../tools/iris_vendor_bridge.hpp
	$(HIPCC) -O2 -std=c++17 -fPIC -shared $< -o $@
run: all
	./generate_dataset --iterations 100 --universe 1000 -k 1000
	IRIS_ARCHS=HIP ./main --input out_asc_uniform_1000_1000_100.bin --output out_asc.txt $(RUN_ARGS)
	IRIS_ARCHS=HIP ./main --input out_desc_uniform_1000_1000_100.bin --output out_desc.txt $(RUN_ARGS)
clean:
	rm -f main generate_dataset libhecbench_vendor.so *.o
