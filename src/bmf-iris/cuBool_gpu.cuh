#ifndef cuBool_GPU_CUH
#define cuBool_GPU_CUH

#include <vector>
#include <iostream>
#include <sstream>
#include <limits>
#include <type_traits>
#include <omp.h>
#include <hip/hip_runtime.h>

#include "helper/rngpu.hpp"
#include "helper/cuda_helpers.cuh"

#include "config.h"  // WARPSPERBLOCK = 16
#include "io_and_allocation.hpp"
#include "dmem_bridge.hpp"
using std::min;

using std::ostringstream;
using std::vector;

template<typename factor_t = uint32_t>
class cuBool
{
  public:
    using factor_matrix_t = vector<factor_t>;
    using bit_vector_t = uint32_t;
    using bit_matrix_t = vector<bit_vector_t>;
    using index_t = uint32_t;
    using error_t = float;
    using cuBool_config = ::cuBool_config<index_t, error_t>;

  private:
    struct factor_handler {
      factor_t *d_A;
      factor_t *d_B;
      error_t *distance_;
      error_t *d_distance_;
      uint8_t factorDim_ = 20;
      size_t lineSize_ = 1;
      bool initialized_ = false;
    };

  public:
    cuBool(const bit_matrix_t& C,
           const index_t height,
           const index_t width,
           const float density,
           const size_t numActiveExperriments = 1)
    {
      std::cout << "~~~ GPU cuBool ~~~" << std::endl;

      int device_id = 0;
      hipDeviceProp_t prop;
      hipGetDeviceProperties(&prop, device_id);

      std::cout << "Using device " << device_id << ": " << prop.name << std::endl;

      max_parallel_lines_ = prop.multiProcessorCount * WARPSPERBLOCK;

      height_ = height;
      width_ = width;

      density_ = density;
      inverse_density_ = 1 / density;

      if(std::is_same<factor_t, uint32_t>::value)
        lineSize_padded_ = 1;
      else if(std::is_same<factor_t, float>::value)
        lineSize_padded_ = 32;

      omp_set_num_threads(numActiveExperriments);
      activeExperiments.resize(numActiveExperriments);
      bestFactors = {};

      allocate();
      std::cout << "cuBool allocation complete." << std::endl;
      std::cout << "Matrix dimensions:\t" << height_ << "x" << width_ << std::endl;

      resetBest();

      initializeMatrix(C);

      if(initialized_)
        std::cout << "cuBool initialization complete." << std::endl;
      else
        exit(1);
      std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -" << std::endl;
    }

  private:
    // allocate memory for matrix, factors and distances
    bool allocate() {
      size_t lineBytes_padded = sizeof(factor_t) * lineSize_padded_;

      for(auto& e : activeExperiments) {
        irisAllocate(&e.d_A, lineBytes_padded * height_); CUERR
        irisAllocate(&e.d_B, lineBytes_padded * width_); CUERR
        hostAllocate(&e.distance_, sizeof(error_t)); CUERR
        irisAllocate(&e.d_distance_, sizeof(error_t)); CUERR
      }

      hostAllocate(&bestFactors.d_A, lineBytes_padded * height_); CUERR
      hostAllocate(&bestFactors.d_B, lineBytes_padded * width_); CUERR
      hostAllocate(&bestFactors.distance_, sizeof(error_t)); CUERR

      index_t height_C = SDIV(height_, 32);
      width_C_padded_ = SDIV(width_, 32) * 32;
      irisAllocate(&d_C, sizeof(bit_vector_t) * height_C * width_C_padded_); CUERR

      return true;
    }

  public:
    ~cuBool() {
      irisFree(d_C);
      for(auto& e : activeExperiments) {
        irisFree(e.d_A);
        irisFree(e.d_B);
        free(e.distance_);
        irisFree(e.d_distance_);
      }
      free(bestFactors.d_A);
      free(bestFactors.d_B);
      free(bestFactors.distance_);
    }

    bool initializeMatrix(const bit_matrix_t& C) {
      if( SDIV(height_,32) * width_ != C.size()) {
        std::cerr << "cuBool construction: Matrix dimension mismatch." << std::endl;
        return false;
      }

      index_t height_C = SDIV(height_, 32);
      width_C_padded_ = SDIV(width_, 32) * 32;

      // emulate 2D copy with 1D copy naively
      // Copies a matrix ("height_C" rows of "sizeof(bit_vector_t) * width_" bytes each)
      // from C.data() to d_C
      // destination pitch and source pitch are the widths in memory in bytes of the 2D arrays

      for (unsigned int i = 0; i < height_C; i++) {
        irisCopy(d_C + i * width_C_padded_, C.data() + i * width_,
                   sizeof(bit_vector_t) * width_, HostToDevice);
        CUERR;
      }

      /*
      hipMemcpy2D(d_C, sizeof(bit_vector_t) * width_C_padded_,
          C.data(), sizeof(bit_vector_t) * width_,
          sizeof(bit_vector_t) * width_,
          height_C,
          HostToDevice); CUERR
       */

        return initialized_ = true;
    }

    bool resetBest() {
      *bestFactors.distance_ = std::numeric_limits<error_t>::max();
      bestFactors.initialized_ = false;
      return true;
    }

  private:
    void calculateDistance(const factor_handler& handler, const error_t weight = 1, const int stream = 0) {
      irisSet(handler.d_distance_, 0, sizeof(error_t), stream);

      {
auto* memory0=irisArgument(handler.d_A,iris_r);
auto* memory1=irisArgument(handler.d_B,iris_r);
auto* memory2=irisArgument(d_C,iris_r);
uint32_t scalar3=height_;
uint32_t scalar4=width_;
uint32_t scalar5=width_C_padded_;
uint8_t scalar6=handler.factorDim_;
float scalar7=weight;
auto* memory8=irisArgument(handler.d_distance_,iris_rw);
void* params[]={memory0,memory1,memory2,&scalar3,&scalar4,&scalar5,&scalar6,&scalar7,memory8};int info[]={iris_r,iris_r,iris_r,sizeof(scalar3),sizeof(scalar4),sizeof(scalar5),sizeof(scalar6),sizeof(scalar7),iris_rw};size_t local=WARPSPERBLOCK*32,global=(SDIV(height_, WARPSPERBLOCK))*local;iris::Task task;task.kernel("iris_computeDistanceRowsShared",1,nullptr,&global,&local,9,params,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("task computeDistanceRowsShared");}

      irisCopy(handler.distance_, handler.d_distance_, sizeof(error_t), DeviceToHost, stream);
       CUERR
    }

  public:
    // initialize factors with custom Initializer function
    template<class Initializer>
    bool initializeFactors(const size_t activeId, const uint8_t factorDim, Initializer&& initilize, const int stream = 0) {
      auto& handler = activeExperiments[activeId];

      handler.factorDim_ = factorDim;

      if(std::is_same<factor_t, uint32_t>::value) {
        handler.lineSize_ = 1;
      }
      else if(std::is_same<factor_t, float>::value) {
        handler.lineSize_ = handler.factorDim_;
      }

      initilize(handler);

      *handler.distance_ = -1;

      return handler.initialized_ = true;
    }

    // initialize factors as copy of host vectors
    bool initializeFactors(const size_t activeId,
        const factor_matrix_t& A,
        const factor_matrix_t& B,
        const uint8_t factorDim,
        const int stream = 0)
    {
      return initializeFactors(activeId, factorDim, [&,this](factor_handler& handler){
        if( A.size() != height_ * handler.lineSize_ || B.size() != width_ * handler.lineSize_) {
          std::cerr << "cuBool initialization: Factor dimension mismatch." << std::endl;
          return false;
        }

        size_t lineBytes = sizeof(factor_t) * handler.lineSize_;
        //size_t lineBytes_padded = sizeof(factor_t) * lineSize_padded_;

        // emulate 2D copy with 1D copy naively
        for (int i = 0; i < height_; i++) {
          irisCopy(handler.d_A + i*lineSize_padded_,
                          A.data() + i*handler.lineSize_,
                          lineBytes,
                          HostToDevice,
                          stream);
        }

        for (int i = 0; i < width_; i++) {
          irisCopy(handler.d_B + i*lineSize_padded_,
                          B.data() + i*handler.lineSize_,
                          lineBytes,
                          HostToDevice,
                          stream);
        }

        /*
        hipMemcpy2DAsync(handler.d_A, lineBytes_padded,
            A.data(), lineBytes,
            lineBytes,
            height_,
            HostToDevice,
            stream);

        hipMemcpy2DAsync(handler.d_B, lineBytes_padded,
            B.data(), lineBytes,
            lineBytes,
            width_,
            HostToDevice,
            stream);
        */
      });
    }

    // initialize factors on device according to INITIALIZATIONMODE
    bool initializeFactors(const size_t activeId, const uint8_t factorDim, uint32_t seed, const int stream = 0) {
      return initializeFactors(activeId, factorDim, [&,this](factor_handler& handler){
        float threshold = getInitChance(density_, handler.factorDim_);

        {
auto* memory0=irisArgument(handler.d_A,iris_rw);
uint32_t scalar1=height_;
uint8_t scalar2=handler.factorDim_;
uint32_t scalar3=seed;
float scalar4=threshold;
void* params[]={memory0,&scalar1,&scalar2,&scalar3,&scalar4};int info[]={iris_rw,sizeof(scalar1),sizeof(scalar2),sizeof(scalar3),sizeof(scalar4)};size_t local=WARPSPERBLOCK*32,global=(SDIV(height_, WARPSPERBLOCK*32/lineSize_padded_))*local;iris::Task task;task.kernel("iris_initFactor",1,nullptr,&global,&local,5,params,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("task initFactor");}

        seed += height_;
          {
auto* memory0=irisArgument(handler.d_B,iris_rw);
uint32_t scalar1=width_;
uint8_t scalar2=handler.factorDim_;
uint32_t scalar3=seed;
float scalar4=threshold;
void* params[]={memory0,&scalar1,&scalar2,&scalar3,&scalar4};int info[]={iris_rw,sizeof(scalar1),sizeof(scalar2),sizeof(scalar3),sizeof(scalar4)};size_t local=WARPSPERBLOCK*32,global=(SDIV(width_, WARPSPERBLOCK*32/lineSize_padded_))*local;iris::Task task;task.kernel("iris_initFactor",1,nullptr,&global,&local,5,params,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("task initFactor");}
      });
    }

    bool verifyDistance(const size_t activeId, const int weight = 1) {
      auto& handler = activeExperiments[activeId];

      if(!initialized_) {
        std::cerr << "cuBool matrix not initialized." << std::endl;
        return false;
      }

      error_t* distance_proof;
      error_t* d_distance_proof;

      hostAllocate(&distance_proof, sizeof(error_t)); CUERR
      irisAllocate(&d_distance_proof, sizeof(error_t)); CUERR
      irisSet(d_distance_proof, 0, sizeof(error_t)); CUERR

      {
auto* memory0=irisArgument(handler.d_A,iris_r);
auto* memory1=irisArgument(handler.d_B,iris_r);
auto* memory2=irisArgument(d_C,iris_r);
uint32_t scalar3=height_;
uint32_t scalar4=width_;
uint32_t scalar5=width_C_padded_;
uint8_t scalar6=handler.factorDim_;
float scalar7=weight;
auto* memory8=irisArgument(d_distance_proof,iris_rw);
void* params[]={memory0,memory1,memory2,&scalar3,&scalar4,&scalar5,&scalar6,&scalar7,memory8};int info[]={iris_r,iris_r,iris_r,sizeof(scalar3),sizeof(scalar4),sizeof(scalar5),sizeof(scalar6),sizeof(scalar7),iris_rw};size_t local=WARPSPERBLOCK*32,global=(SDIV(height_, WARPSPERBLOCK))*local;iris::Task task;task.kernel("iris_computeDistanceRowsShared",1,nullptr,&global,&local,9,params,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("task computeDistanceRowsShared");}

       CUERR

      irisCopy(distance_proof, d_distance_proof, sizeof(error_t), DeviceToHost); CUERR

      bool equal = *handler.distance_ == *distance_proof;
      if(!equal) {
        std::cout << "----- !Distances differ! -----\n";
        std::cout << "Running distance:  " << *handler.distance_ << "\n";
        std::cout << "Real distance:     " << *distance_proof << std::endl;
      } else {
        std::cout << "Distance verified" << std::endl;
      }

      free(distance_proof);
      irisFree(d_distance_proof);
      return equal;
    }

    void getFactors(const size_t activeId, factor_matrix_t& A, factor_matrix_t& B, const int stream = 0) const {
      auto& handler = activeExperiments[activeId];

      if(!handler.initialized_) {
        std::cerr << "Factors in slot " << activeId << " not initialized." << std::endl;
        return;
      }

      size_t lineBytes = sizeof(factor_t) * handler.lineSize_;

      A.resize(height_);

      for (int i = 0; i < height_; i++) {
        irisCopy(A.data() + i*handler.lineSize_,
                        handler.d_A + i*lineSize_padded_,
                        lineBytes,
                        DeviceToHost,
                        stream);
      }

      for (int i = 0; i < width_; i++) {
        irisCopy(B.data() + i*handler.lineSize_,
                        handler.d_B + i*lineSize_padded_,
                        lineBytes,
                        DeviceToHost,
                        stream);
      }

      /*
      hipMemcpy2DAsync(A.data(), lineBytes,
          handler.d_A, lineBytes_padded,
          lineBytes,
          height_,
          DeviceToHost,
          stream);

      B.resize(width_);
      hipMemcpy2DAsync(B.data(), lineBytes,
          handler.d_B, lineBytes_padded,
          lineBytes,
          width_,
          DeviceToHost,
          stream);
      */
    }

    error_t getDistance(const size_t activeId) const {
      auto& handler = activeExperiments[activeId];

      if(!handler.initialized_) {
        std::cerr << "Factors in slot " << activeId << " not initialized." << std::endl;
        return -1;
      }
      return *handler.distance_;
    }

    void getBestFactors(factor_matrix_t& A, factor_matrix_t& B, const int stream = 0) const {
      if(!bestFactors.initialized_) {
        std::cerr << "Best result not initialized." << std::endl;
        return;
      }

      size_t lineBytes = sizeof(factor_t) * bestFactors.lineSize_;

      A.resize(height_);
      irisCopy(A.data(),
          bestFactors.d_A,
          lineBytes * height_,
          HostToHost,
          stream);

      B.resize(width_);
      irisCopy(B.data(),
          bestFactors.d_B,
          lineBytes * width_,
          HostToHost,
          stream);
    }

    error_t getBestDistance() const {
      if(!bestFactors.initialized_) {
        std::cerr << "Best result not initialized." << std::endl;
        return -1;
      }
      return *bestFactors.distance_;
    }

    void runMultiple(const size_t numExperiments, const cuBool_config& config) {
      finalDistances.resize(numExperiments);

      fast_kiss_state32_t state = get_initial_fast_kiss_state32(config.seed);

      #pragma omp parallel for schedule(dynamic,1) shared(state)
      for(size_t i=0; i<numExperiments; ++i) {
        unsigned id = omp_get_thread_num();
        auto config_i = config;
        uint32_t seed;
        #pragma omp critical(kiss)
        seed = fast_kiss32(state);
        #pragma omp critical(kiss)
        config_i.seed = fast_kiss32(state);
        #pragma omp critical(cout)
        std::cout << "Starting run " << i << " in slot " << id << " with seed " << config_i.seed << std::endl;
        initializeFactors(id, config_i.factorDim, seed);
        finalDistances[i] = run(id, config_i);
      }
    }

    float run(const size_t activeId, const cuBool_config& config, const int stream = 0) {
      auto& handler = activeExperiments[activeId];

      if(!initialized_) {
        std::cerr << "cuBool matrix not initialized." << std::endl;
        return -1;
      }

      if(!handler.initialized_) {
        std::cerr << "cuBool factors in slot " << activeId << " not initialized." << std::endl;
        return -1;
      }

      ostringstream out;

      calculateDistance(handler, config.weight, stream);

      if(config.verbosity > 0) {
        out << "\tStart distance for slot " << activeId
            << "\tabs_err: " << *handler.distance_
            << "\trel_err: " << float(*handler.distance_) / height_ / width_
            << '\n';
      }

      index_t linesAtOnce = SDIV(config.linesAtOnce, WARPSPERBLOCK) * WARPSPERBLOCK;
      if(config.loadBalance) {
        linesAtOnce = linesAtOnce / max_parallel_lines_ * max_parallel_lines_;
        if (!linesAtOnce) linesAtOnce = max_parallel_lines_;
      }

      if(config.verbosity > 1) {
        out << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n";
        out << "- - - - Starting " << config.maxIterations
            << " GPU iterations, changing " << linesAtOnce
            << " lines each time\n";
        out << "- - - - Showing error every " << config.distanceShowEvery
          << " steps\n";
        if(config.tempStart > 0) {
          out << "- - - - Start temperature " << config.tempStart
              << " multiplied by " << config.reduceFactor
              << " every " << config.reduceStep
              << " steps\n";
          out << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
              << std::endl;
        }
      }

      fast_kiss_state32_t state = get_initial_fast_kiss_state32(config.seed);
      float temperature = config.tempStart;
      float weight = config.weight;
      size_t iteration = 0;
      size_t stuckIterations = 0;
      auto distancePrev = *handler.distance_;
      size_t syncStep = 100;
      while(
          *handler.distance_ > config.distanceThreshold &&
          iteration++ < config.maxIterations
          && temperature > config.tempEnd
          && stuckIterations < config.stuckIterationsBeforeBreak) {

        // Change rows
        index_t lineToBeChanged = (fast_kiss32(state) % height_) / WARPSPERBLOCK * WARPSPERBLOCK;
        uint32_t gpuSeed = fast_kiss32(state) + iteration;

        {
auto* memory0=irisArgument(handler.d_A,iris_rw);
auto* memory1=irisArgument(handler.d_B,iris_r);
auto* memory2=irisArgument(d_C,iris_r);
uint32_t scalar3=height_;
uint32_t scalar4=width_;
uint32_t scalar5=width_C_padded_;
uint8_t scalar6=handler.factorDim_;
uint32_t scalar7=lineToBeChanged;
auto* memory8=irisArgument(handler.d_distance_,iris_rw);
uint32_t scalar9=gpuSeed;
float scalar10=temperature/10;
float scalar11=config.flipManyChance;
uint32_t scalar12=config.flipManyDepth;
float scalar13=weight;
void* params[]={memory0,memory1,memory2,&scalar3,&scalar4,&scalar5,&scalar6,&scalar7,memory8,&scalar9,&scalar10,&scalar11,&scalar12,&scalar13};int info[]={iris_rw,iris_r,iris_r,sizeof(scalar3),sizeof(scalar4),sizeof(scalar5),sizeof(scalar6),sizeof(scalar7),iris_rw,sizeof(scalar9),sizeof(scalar10),sizeof(scalar11),sizeof(scalar12),sizeof(scalar13)};size_t local=WARPSPERBLOCK*32,global=(SDIV(min(linesAtOnce, height_), WARPSPERBLOCK))*local;iris::Task task;task.kernel("iris_vectorMatrixMultCompareRowWarpShared",1,nullptr,&global,&local,14,params,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("task vectorMatrixMultCompareRowWarpShared");}
        //  CUERR

        // Change cols
        lineToBeChanged = (fast_kiss32(state) % width_) / WARPSPERBLOCK * WARPSPERBLOCK;
        gpuSeed = fast_kiss32(state) + iteration;

        {
auto* memory0=irisArgument(handler.d_A,iris_r);
auto* memory1=irisArgument(handler.d_B,iris_rw);
auto* memory2=irisArgument(d_C,iris_r);
uint32_t scalar3=height_;
uint32_t scalar4=width_;
uint32_t scalar5=width_C_padded_;
uint8_t scalar6=handler.factorDim_;
uint32_t scalar7=lineToBeChanged;
auto* memory8=irisArgument(handler.d_distance_,iris_rw);
uint32_t scalar9=gpuSeed;
float scalar10=temperature/10;
float scalar11=config.flipManyChance;
uint32_t scalar12=config.flipManyDepth;
float scalar13=weight;
void* params[]={memory0,memory1,memory2,&scalar3,&scalar4,&scalar5,&scalar6,&scalar7,memory8,&scalar9,&scalar10,&scalar11,&scalar12,&scalar13};int info[]={iris_r,iris_rw,iris_r,sizeof(scalar3),sizeof(scalar4),sizeof(scalar5),sizeof(scalar6),sizeof(scalar7),iris_rw,sizeof(scalar9),sizeof(scalar10),sizeof(scalar11),sizeof(scalar12),sizeof(scalar13)};size_t local=WARPSPERBLOCK*32,global=(SDIV(min(linesAtOnce, width_), WARPSPERBLOCK))*local;iris::Task task;task.kernel("iris_vectorMatrixMultCompareColWarpShared",1,nullptr,&global,&local,14,params,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("task vectorMatrixMultCompareColWarpShared");}
         CUERR

        if(iteration % syncStep == 0) {
          irisCopy(handler.distance_, handler.d_distance_, sizeof(error_t), DeviceToHost, stream);
           CUERR

          if(*handler.distance_ == distancePrev)
            stuckIterations += syncStep;
          else
            stuckIterations = 0;
          distancePrev = *handler.distance_;
        }

        if(config.verbosity > 1 && iteration % config.distanceShowEvery == 0) {
          out << "Iteration: " << iteration
              << "\tabs_err: " << *handler.distance_
              << "\trel_err: " << float(*handler.distance_) / height_ / width_
              << "\ttemp: " << temperature;
          out << std::endl;
        }
        if(iteration % config.reduceStep == 0) {
          temperature *= config.reduceFactor;
          if(weight > 1)
            weight *= config.reduceFactor;
          if(weight < 1)
            weight = 1;
        }
      }

      if(config.verbosity > 0) {
        out << "\tBreak condition for slot " << activeId << ":\t";
        if (!(iteration < config.maxIterations))
          out << "Reached iteration limit: " << config.maxIterations;
        if (!(*handler.distance_ > config.distanceThreshold))
          out << "Distance below threshold: " << config.distanceThreshold;
        if (!(temperature > config.tempEnd))
          out << "Temperature below threshold";
        if (!(stuckIterations < config.stuckIterationsBeforeBreak))
          out << "Stuck for " << stuckIterations << " iterations";
        out << " after " << iteration << " iterations.\n";
      }

      // use hamming distance for final judgement
      calculateDistance(handler, 1, stream);

      if(config.verbosity > 0) {
        out << "\tFinal distance for slot " << activeId
            << "\tabs_err: " << *handler.distance_
            << "\trel_err: " << float(*handler.distance_) / height_ / width_
            << std::endl;
      }

      if(*handler.distance_ < *bestFactors.distance_) {

        #pragma omp critical
        if(*handler.distance_ < *bestFactors.distance_) {
          if(config.verbosity > 0) {
            out << "\tResult is better than previous best. Copying to host." << std::endl;
          }

          *bestFactors.distance_ = *handler.distance_;
          bestFactors.lineSize_ = handler.lineSize_;
          bestFactors.factorDim_ = handler.factorDim_;

          size_t lineBytes = sizeof(factor_t) * handler.lineSize_;

          for (unsigned int i = 0; i < height_; i++) {
            irisCopy(bestFactors.d_A + i*handler.lineSize_,
                            handler.d_A + i*lineSize_padded_,
                            lineBytes,
                            DeviceToHost,
                            stream);
          }

          for (unsigned int i = 0; i < width_; i++) {
            irisCopy(bestFactors.d_B + i*handler.lineSize_,
                            handler.d_B + i*lineSize_padded_,
                            lineBytes,
                            DeviceToHost,
                            stream);
          }
/*
          hipMemcpy2DAsync(bestFactors.d_A, lineBytes,
              handler.d_A, lineBytes_padded,
              lineBytes,
              height_,
              DeviceToHost,
              stream);

          hipMemcpy2DAsync(bestFactors.d_B, lineBytes,
              handler.d_B, lineBytes_padded,
              lineBytes,
              width_,
              DeviceToHost,
              stream);
*/

          bestFactors.initialized_ = true;
        }
      }

      #pragma omp critical
      std::cout << out.str();

      return float(*handler.distance_) / height_ / width_;
    }

    const vector<float>& getDistances() const {
      return finalDistances;
    }

  private:
    bool initialized_ = false;
    bit_vector_t *d_C;
    float density_;
    int inverse_density_;
    index_t height_;
    index_t width_;
    index_t width_C_padded_;
    size_t lineSize_padded_;
    int max_parallel_lines_;
    factor_handler bestFactors;
    vector<factor_handler> activeExperiments;
    vector<float> finalDistances;
};

#endif
