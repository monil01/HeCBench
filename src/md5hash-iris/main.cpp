#include "md5_common.h"
#include "dmem.hpp"
void FindKeyWithDigest_GPU(
  const unsigned int searchDigest[4],
  const int byteLength,
  const int valsPerByte,
  int *foundIndex,
  unsigned char foundKey[8],
  unsigned int foundDigest[4])
{
  int *d_foundIndex;
  unsigned char *d_foundKey;
  unsigned int *d_foundDigest;
  port::allocate((void**)&d_foundIndex, sizeof(int));
  port::allocate((void**)&d_foundKey, sizeof(unsigned char) * 8);
  port::allocate((void**)&d_foundDigest, sizeof(unsigned int) * 4);
  port::copy(d_foundKey, foundKey, sizeof(unsigned char) * 8, port::h2d);
  port::copy(d_foundIndex, foundIndex, sizeof(int), port::h2d);
  port::copy(d_foundDigest, foundDigest, sizeof(unsigned int) * 4, port::h2d);

  //
  // calculate work thread shape
  //
  int keyspace = FindKeyspaceSize(byteLength, valsPerByte);
  size_t nthreads = 256;
  size_t nblocks  = ceil((double(keyspace) / double(valsPerByte)) / double(nthreads));

  unsigned int searchDigest0 = searchDigest[0];
  unsigned int searchDigest1 = searchDigest[1];
  unsigned int searchDigest2 = searchDigest[2];
  unsigned int searchDigest3 = searchDigest[3];

  port::launch("md5hash_kernel",dim3(nblocks),dim3(nthreads),
      d_foundIndex, d_foundKey, d_foundDigest,
      keyspace, byteLength, valsPerByte,
      searchDigest0, searchDigest1, searchDigest2, searchDigest3);

  port::copy(foundKey, d_foundKey, sizeof(unsigned char) * 8, port::d2h);
  port::copy(foundIndex, d_foundIndex, sizeof(int), port::d2h);
  port::copy(foundDigest, d_foundDigest, sizeof(unsigned int) * 4, port::d2h);
  port::release(d_foundDigest);
  port::release(d_foundIndex);
  port::release(d_foundKey);
}


// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the MD5 Hash benchmark
//
// Arguments:
//   dev: the opencl device id to use for the benchmark
//   ctx: the opencl context to use for the benchmark
//   queue: the opencl command queue to issue commands to
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Jeremy Meredith
// Creation: July 23, 2014
//
// Modifications:
//
// ****************************************************************************

int main(int argc, char** argv)
{
  port::Session session(&argc, &argv);
  if (argc != 3) {
    printf("Usage: %s <offload>  <passes>\n"
        "offload: 1 for target device offload\n"
        "passes: Number of run iterations\n" , argv[0]);
    return 1;
  }

  // cpu when offload = 0 ; gpu when offload != 0
  int offload = atoi(argv[1]);

  // number of iterations
  int passes = atoi(argv[2]);

  //
  bool verbose = true;

  for (int size = 1; size <= 4; size++) {
    //
    // Determine the shape/size of key space
    //
    const int sizes_byteLength[]  = { 7,  5,  6,  5};
    const int sizes_valsPerByte[] = {10, 35, 25, 70};

    const int byteLength = sizes_byteLength[size-1];
    const int valsPerByte = getenv("HECBENCH_MD5_VALUES") ? atoi(getenv("HECBENCH_MD5_VALUES")) : sizes_valsPerByte[size-1];

    char atts[1024];
    sprintf(atts, "%dx%d", byteLength, valsPerByte);

    if (verbose)
      cout << "Searching keys of length " << byteLength << " bytes "
           << "and " << valsPerByte << " values per byte" << endl;

    const int keyspace = FindKeyspaceSize(byteLength, valsPerByte);
    if (keyspace < 0)
    {
      cerr << "Error: more than 2^31 bits of entropy is unsupported.\n";
      return -1;
    }

    if (byteLength > 7)
    {
      cerr << "Error: more than 7 byte key length is unsupported.\n";
      return -1;
    }

    if (verbose)
      cout << "|keyspace| = " << keyspace << " ("<<int(keyspace/1e6)<<"M)" << endl;

    //
    // Choose a random key from the keyspace, and calculate its hash.
    //
    srandom(12345);

    for (int pass = 0 ; pass < passes ; ++pass)
    {
      int randomIndex = random() % keyspace;;
      unsigned char randomKey[8] = {0,0,0,0, 0,0,0,0};
      unsigned int randomDigest[4];
      IndexToKey(randomIndex, byteLength, valsPerByte, randomKey);
      md5_2words((unsigned int*)randomKey, byteLength, randomDigest);

      if (verbose)
      {
        cout << endl;
        cout << "--- iteration " << pass << " ---" << endl;
        cout << "Looking for random key:" << endl;
        cout << " randomIndex = " << randomIndex << endl;
        cout << " randomKey   = 0x" << AsHex(randomKey, 8/*byteLength*/) << endl;
        cout << " randomDigest= " << AsHex((unsigned char*)randomDigest, 16) << endl;
      }

      //
      // Use the GPU to brute force search the keyspace for this key.
      //
      unsigned int foundDigest[4] = {0,0,0,0};
      int foundIndex = -1;
      unsigned char foundKey[8] = {0,0,0,0, 0,0,0,0};

      auto start = std::chrono::steady_clock::now();
      if (offload == 0)
      {
        FindKeyWithDigest_CPU(randomDigest, byteLength, valsPerByte,
            &foundIndex, foundKey, foundDigest);
      }
      else
      {
        FindKeyWithDigest_GPU(randomDigest, byteLength, valsPerByte,
            &foundIndex, foundKey, foundDigest);
      }
      auto end = std::chrono::steady_clock::now();
      auto t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

      //
      // Calculate the rate and add it to the results
      //
      double rate = double(keyspace) / (double(t)/1000) / 1.e9;

      if (verbose)
      {
        cout << "time = " << t << " ms, rate = " << rate << " GHash/sec\n";
      }

      //
      // Double check everything matches (index, key, hash).
      //
      if (foundIndex < 0)
      {
        cerr << "\nERROR: could not find a match.\n";
        rate = FLT_MAX;
      }
      else if (foundIndex != randomIndex)
      {
        cerr << "\nERROR: mismatch in key index found.\n";
        rate = FLT_MAX;
      }
      else if (foundKey[0] != randomKey[0] ||
          foundKey[1] != randomKey[1] ||
          foundKey[2] != randomKey[2] ||
          foundKey[3] != randomKey[3] ||
          foundKey[4] != randomKey[4] ||
          foundKey[5] != randomKey[5] ||
          foundKey[6] != randomKey[6] ||
          foundKey[7] != randomKey[7])
      {
        cerr << "\nERROR: mismatch in key value found.\n";
        rate = FLT_MAX;
      }
      else if (foundDigest[0] != randomDigest[0] ||
          foundDigest[1] != randomDigest[1] ||
          foundDigest[2] != randomDigest[2] ||
          foundDigest[3] != randomDigest[3])
      {
        cerr << "\nERROR: mismatch in digest of key.\n";
        rate = FLT_MAX;
      }
      else
      {
        if (verbose)
          cout << endl << "Successfully found match (index, key, hash):" << endl;
      }

      if (verbose)
      {
        cout << " foundIndex  = " << foundIndex << endl;
        cout << " foundKey    = 0x" << AsHex(foundKey, 8/*byteLength*/) << endl;
        cout << " foundDigest = " << AsHex((unsigned char*)foundDigest, 16) << endl;
        cout << endl;
      }
      cout << ((rate == FLT_MAX) ? "FAIL" : "PASS") << endl;
    }
  }

  return 0;
}
