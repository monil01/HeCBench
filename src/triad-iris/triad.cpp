#include <math.h>
#include <string.h>
#include "runtime.h"

#include "OptionParser.h"
#include "Timer.h"
#include "Utility.h"

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: December 15, 2009
//
// Modifications:
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
  ;
}

// ****************************************************************************
// Function: triad
//
// Purpose:
//   A simple vector addition kernel
//   C = A + s*B
//
// Arguments:
//   A,B - input vectors
//   C - output vectors
//   s - scalar
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: December 15, 2009
//
// Modifications:
//
// ****************************************************************************


// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Implements the Stream Triad benchmark in CUDA.  This benchmark
//   is designed to test CUDA's overall data transfer speed. It executes
//   a vector addition operation with no temporal reuse. Data is read
//   directly from the global memory. This implementation tiles the input
//   array and pipelines the vector addition computation with
//   the data download for the next tile. However, since data transfer from
//   host to device is much more expensive than the simple vector computation,
//   data transfer operations should completely dominate the execution time.
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser (contains input parameters)
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: December 15, 2009
//
// Modifications:
//
// ****************************************************************************
void RunBenchmark(OptionParser &op)
{
  const bool verbose = op.getOptionBool("verbose");
  const int n_passes = op.getOptionInt("passes");

  // 256k through 8M bytes
  const int nSizes = 9;
  const int blockSizes[] = { 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 };
  const int memSize = 16384;
  const int numMaxFloats = 1024 * memSize / sizeof(float);
  const int halfNumFloats = numMaxFloats / 2;

  // Create some host memory pattern
  srand48(8650341L);
  float *h_mem;
  h_mem = (float*)malloc(sizeof(float) * numMaxFloats);

  // Allocate some device memory
  Storage* d_memA0, *d_memB0, *d_memC0;
  d_memA0 = new Storage(blockSizes[nSizes - 1] * 1024);
  d_memB0 = new Storage(blockSizes[nSizes - 1] * 1024);
  d_memC0 = new Storage(blockSizes[nSizes - 1] * 1024);

  Storage* d_memA1, *d_memB1, *d_memC1;
  d_memA1 = new Storage(blockSizes[nSizes - 1] * 1024);
  d_memB1 = new Storage(blockSizes[nSizes - 1] * 1024);
  d_memC1 = new Storage(blockSizes[nSizes - 1] * 1024);

  float scalar = 1.75f;

  const int blockSize = 128;

  // Step through sizes forward
  for (int i = 0; i < nSizes; ++i)
  {
    int elemsInBlock = blockSizes[i] * 1024 / sizeof(float);
    for (int j = 0; j < halfNumFloats; ++j)
      h_mem[j] = h_mem[halfNumFloats + j] = (float) (drand48() * 10.0);

    // Copy input memory to the device
    if (verbose) {
      cout << ">> Executing Triad with vectors of length "
        << numMaxFloats << " and block size of "
        << elemsInBlock << " elements." << "\n";
      cout << "Block: " << blockSizes[i] << "KB" << "\n";
    }

    // start submitting blocks of data of size elemsInBlock
    // overlap the computation of one block with the data
    // download for the next block and the results upload for
    // the previous block
    int crtIdx = 0;
    int gridSize = elemsInBlock / blockSize;

    
    
    

    int TH = Timer::Start();

    // Number of passes. Use a large number for stress testing.
    // A small value is sufficient for computing sustained performance.
    for (int pass = 0; pass < n_passes; ++pass)
    {
      copy_to(d_memA0, h_mem, blockSizes[i] * 1024);
      copy_to(d_memB0, h_mem, blockSizes[i] * 1024);

      launch_triad(d_memA0, d_memB0, d_memC0, scalar, elemsInBlock);

      if (elemsInBlock < numMaxFloats)
      {
        // start downloading data for next block
        copy_to(d_memA1, h_mem + elemsInBlock, blockSizes[i]
            * 1024);
        copy_to(d_memB1, h_mem + elemsInBlock, blockSizes[i]
            * 1024);
      }

      int blockIdx = 1;
      unsigned int currStream = 1;
      while (crtIdx < numMaxFloats)
      {
        currStream = blockIdx & 1;
        // Start copying back the answer from the last kernel
        if (currStream)
        {
          copy_from(h_mem + crtIdx, d_memC0, elemsInBlock * sizeof(float));
        }
        else
        {
          copy_from(h_mem + crtIdx, d_memC1, elemsInBlock
              * sizeof(float));
        }

        crtIdx += elemsInBlock;

        if (crtIdx < numMaxFloats)
        {
          // Execute the kernel
          if (currStream)
          {
            launch_triad(d_memA1, d_memB1, d_memC1, scalar, elemsInBlock);
          }
          else
          {
            launch_triad(d_memA0, d_memB0, d_memC0, scalar, elemsInBlock);
          }
        }

        if (crtIdx+elemsInBlock < numMaxFloats)
        {
          // Download data for next block
          if (currStream)
          {
            copy_to(d_memA0, h_mem+crtIdx+elemsInBlock, blockSizes[i]*1024);
            copy_to(d_memB0, h_mem+crtIdx+elemsInBlock, blockSizes[i]*1024);
          }
          else
          {
            copy_to(d_memA1, h_mem+crtIdx+elemsInBlock, blockSizes[i]*1024);
            copy_to(d_memB1, h_mem+crtIdx+elemsInBlock, blockSizes[i]*1024);
          }
        }
        blockIdx += 1;
        currStream = !currStream;
      }
      

    } // for (int pass = 0; pass < n_passes; ++pass)

    double time = Timer::Stop(TH, "thread synchronize");

    double triad = ((double)numMaxFloats*2.0*n_passes) / (time*1e9);
    if (verbose) std::cout << "Average TriadFlops " << triad << " GFLOPS/s\n";

    double bdwth = ((double)numMaxFloats*sizeof(float)*3.0*n_passes)
                   / (time*1000.*1000.*1000.);
    if (verbose) std::cout << "Average TriadBdwth " << bdwth << " GB/s\n";

    // Checking memory for correctness. The two halves of the array
    // should have the same results.
    bool ok = true;
    for (int j=0; j<halfNumFloats; ++j)
    {
      if (h_mem[j] != h_mem[j+halfNumFloats])
      {
        cout << "Error; hostMem[" << j << "]=" << h_mem[j]
          << " is different from its twin element hostMem["
          << (j+halfNumFloats) << "]: "
          << h_mem[j+halfNumFloats] << "stopping check\n";
        ok = false;
        break;
      }
    }

    if (ok)
      cout << "PASS\n";
    else
      cout << "FAIL\n";

    if (const char *path = getenv("HECBENCH_VALIDATION_DUMP")) {
      FILE *f=fopen(path,i==0?"wb":"ab");
      if(!f || fwrite(h_mem,sizeof(float),numMaxFloats,f)!=size_t(numMaxFloats)) exit(4);
      fclose(f);
    }

    // Zero out the test host memory
    for (int j=0; j<numMaxFloats; ++j) h_mem[j] = 0.0f;

    
    
  }

  // Cleanup
  delete d_memA0;
  delete d_memB0;
  delete d_memC0;
  delete d_memA1;
  delete d_memB1;
  delete d_memC1;
  free(h_mem);
}
