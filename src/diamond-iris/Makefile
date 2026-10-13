IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
DMND_PATH ?= ../diamond-sycl
CXX ?= g++
HIPCC ?= hipcc
NVCC ?= nvcc
CXXFLAGS ?= -O3 -std=c++17 -Wall
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -lz -lpthread -ldl
LAUNCHER ?=
RUN_ARGS ?= blastx -q $(DMND_PATH)/long.fastq.gz -d $(DMND_PATH)/ARDB.dmnd -o long_cur.m8 --tmpdir /dev/shm -p1
FRAMEWORK_FILES := src/blast/sm_blosum45.c \
  src/blast/sm_blosum50.c \
  src/blast/sm_blosum62.c \
  src/blast/sm_blosum80.c \
  src/blast/sm_blosum90.c \
  src/blast/sm_pam30.c \
  src/blast/sm_pam70.c \
  src/blast/sm_pam250.c \
  src/run/main.cpp \
  src/basic/config.cpp \
  src/util/tinythread.cpp \
  src/util/compressed_stream.cpp \
  src/basic/score_matrix.cpp \
  src/blast/blast_filter.cpp \
  src/blast/blast_seg.cpp \
  src/data/queries.cpp \
  src/data/reference.cpp \
  src/data/seed_histogram.cpp \
  src/output/daa_record.cpp \
  src/search/search.cpp \
  src/util/command_line_parser.cpp \
  src/util/seq_file_format.cpp \
  src/util/util.cpp \
  src/util/Timer.cpp \
  src/basic/basic.cpp \
  src/dp/floating_sw.cpp \
  src/basic/hssp.cpp \
  src/dp/ungapped_align.cpp \
  src/run/tools.cpp \
  src/dp/greedy_align.cpp \
  src/run/benchmark.cpp \
  src/search/stage2.cpp \
  src/output/output_format.cpp \
  src/output/join_blocks.cpp \
  src/data/frequent_seeds.cpp \
  src/align/query_mapper.cpp \
  src/align/align_target.cpp \
  src/output/blast_tab_format.cpp \
  src/dp/padded_banded_sw.cpp \
  src/dp/needleman_wunsch.cpp \
  src/output/blast_pairwise_format.cpp \
  src/extra/roc.cpp \
  src/dp/comp_based_stats.cpp \
  src/extra/model_sim.cpp \
  src/run/double_indexed.cpp \
  src/search/collision.cpp \
  src/output/sam_format.cpp \
  src/align/align.cpp \
  src/search/setup.cpp \
  src/extra/opt.cpp \
  src/dp/diag_scores.cpp \
  src/data/taxonomy.cpp \
  src/lib/tantan/tantan.cpp \
  src/lib/tantan/tantale.cpp \
  src/dp/swipe.cpp \
  src/dp/banded_sw.cpp \
  src/data/sorted_list.cpp \
  src/data/seed_set.cpp \
  src/util/binary_file.cpp \
  src/output/taxon_format.cpp \
  src/output/view.cpp \
  src/output/output_sink.cpp \
  src/output/target_culling.cpp
FRAMEWORK_OBJECTS := $(addprefix obj/,$(addsuffix .o,$(FRAMEWORK_FILES)))
.PHONY: all run test clean
all: main kernel.hip
main: obj/main.o $(FRAMEWORK_OBJECTS)
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)
obj/main.o: main.cpp iris_support.h Makefile
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@
obj/src/run/main.cpp.o: $(DMND_PATH)/src/run/main.cpp Makefile
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -Dmain=diamond_application_main -MMD -MP -c $< -o $@
obj/%.cpp.o: $(DMND_PATH)/%.cpp Makefile
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@
obj/%.c.o: $(DMND_PATH)/%.c Makefile
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@
kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) -O3 --genco -o $@ $<
kernel.ptx: kernel.cu Makefile
	$(NVCC) -O3 -ptx -o $@ $<
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
test: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main blastx -q $(DMND_PATH)/short.fastq.gz -d $(DMND_PATH)/ARDB.dmnd -o short_cur.m8 --tmpdir /dev/shm -p1
clean:
	rm -f main kernel.hip kernel.ptx obj/main.o obj/main.d $(FRAMEWORK_OBJECTS) $(FRAMEWORK_OBJECTS:.o=.d)
-include obj/main.d $(FRAMEWORK_OBJECTS:.o=.d)
