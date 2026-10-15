IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
SOURCE_DIR ?= ../leukocyte-cuda
MATRIX_DIR ?= $(SOURCE_DIR)/meschach_lib
RUN_ARGS ?= ../data/leukocyte/testfile.avi 100
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I. -I$(SOURCE_DIR) -I$(MATRIX_DIR) -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include -include find_ellipse.h -include verification.h
OUTPUT ?= -DOUTPUT
DENSE = copy err matrixio memory vecop matop pxop submat init otherio machine matlab ivecop version meminfo memstat lufactor bkpfacto chfactor qrfactor solve hsehldr givens update norm hessen symmeig schur svd fft mfunc bdfactor
SPARSE = sparse sprow sparseio spchfctr splufctr spbkp spswap iter0 itersym iternsym
COMPLEX = zmachine zcopy zmatio zmemory zvecop zmatop znorm zfunc zlufctr zsolve zmatlab zhsehldr zqrfctr zgivens zhessen zschur
MATRIX_OBJ = $(addprefix obj/mesh-,$(addsuffix .o,$(DENSE) $(SPARSE) $(COMPLEX)))
MATRIX_LIB = obj/meschach.a
HOST_OBJ = $(addprefix obj/,$(addsuffix .o,avilib track_ellipse find_ellipse misc_math))
all: main kernel.hip
obj:
	mkdir -p obj
obj/mesh-%.o: $(MATRIX_DIR)/%.c | obj
	$(CXX) $(CXXFLAGS) -DHAVE_CONFIG_H -DANSI_C -I$(MATRIX_DIR) -x c++ -c $< -o $@
obj/%.o: $(SOURCE_DIR)/%.c find_ellipse.h verification.h | obj
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OUTPUT) -include find_ellipse.h -x c++ -c $< -o $@
$(MATRIX_LIB): $(MATRIX_OBJ)
	$(AR) rcs $@ $^
main: main.cpp track_ellipse_gpu.cpp find_ellipse.h iris_helpers.h verification.h $(HOST_OBJ) $(MATRIX_LIB)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OUTPUT) main.cpp track_ellipse_gpu.cpp $(HOST_OBJ) $(MATRIX_LIB) -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread -lm
kernel.hip: kernel.hip.cpp
	$(HIPCC) -O3 --genco -o $@ $<
reference: interpolation_reference.cpp obj/find_ellipse.o obj/avilib.o obj/misc_math.o $(MATRIX_LIB)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $^ -o $@ -lm
mgvf-reference: mgvf_reference_driver.cpp track_ellipse_gpu.cpp iris_helpers.h verification.h $(MATRIX_LIB)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -DIRIS_REFERENCE_DRIVER mgvf_reference_driver.cpp track_ellipse_gpu.cpp $(MATRIX_LIB) -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread -lm
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx reference mgvf-reference obj/*.o obj/meschach.a
