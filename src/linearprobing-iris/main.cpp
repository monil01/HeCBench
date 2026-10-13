#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <chrono>
#include "linearprobing.h"

// Create random keys/values in the range [0, kEmpty)
// kEmpty is used to indicate an empty slot
std::vector<KeyValue> generate_random_keyvalues(std::mt19937& rnd, uint32_t numkvs)
{
  std::uniform_int_distribution<uint32_t> dis(0, kEmpty - 1);

  std::vector<KeyValue> kvs;
  kvs.reserve(numkvs);

  for (uint32_t i = 0; i < numkvs; i++)
  {
    uint32_t rand0 = dis(rnd);
    uint32_t rand1 = dis(rnd);
    kvs.push_back(KeyValue{rand0, rand1});
  }

  return kvs;
}

// return numshuffledkvs random items from kvs
std::vector<KeyValue> shuffle_keyvalues(std::mt19937& rnd, std::vector<KeyValue> kvs, uint32_t numshuffledkvs)
{
  std::shuffle(kvs.begin(), kvs.end(), rnd);

  std::vector<KeyValue> shuffled_kvs;
  shuffled_kvs.resize(numshuffledkvs);

  std::copy(kvs.begin(), kvs.begin() + numshuffledkvs, shuffled_kvs.begin());

  return shuffled_kvs;
}

using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;

Time start_timer()
{
  return std::chrono::high_resolution_clock::now();
}

double get_elapsed_time(Time start)
{
  Time end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> d = end - start;
  std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(d);
  return us.count() / 1000.0f;
}

void test_unordered_map(std::vector<KeyValue> insert_kvs, std::vector<KeyValue> delete_kvs)
{
  Time timer = start_timer();

  printf("Timing std::unordered_map...\n");

  {
    std::unordered_map<uint32_t, uint32_t> kvs_map;
    for (auto& kv : insert_kvs)
    {
      kvs_map[kv.key] = kv.value;
    }
    for (auto& kv : delete_kvs)
    {
      auto i = kvs_map.find(kv.key);
      if (i != kvs_map.end())
        kvs_map.erase(i);
    }
  }

  double milliseconds = get_elapsed_time(timer);
  double seconds = milliseconds / 1000.0f;
  printf("Total time for std::unordered_map: %f ms (%f million keys/second)\n",
      milliseconds, kNumKeyValues / seconds / 1000000.0f);
}

bool test_correctness(std::vector<KeyValue>, std::vector<KeyValue>, std::vector<KeyValue>);

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of insert batches> <number of delete batches>\n", argv[0]);
    return 1;
  }
  IRISSession session(&argc,&argv);
  const uint32_t num_insert_batches = atoi(argv[1]);
  const uint32_t num_delete_batches = atoi(argv[2]);

  // To recreate the same random numbers across runs of the program, set seed to a specific
  // number instead of a number from random_device
  std::random_device rd;
  uint32_t seed = 123;
  std::mt19937 rnd(seed);  // mersenne_twister_engine

  printf("Random number generator seed = %u\n", seed);

  printf("Initializing keyvalue pairs with random numbers...\n");

  std::vector<KeyValue> insert_kvs = generate_random_keyvalues(rnd, kNumKeyValues);
  std::vector<KeyValue> delete_kvs = shuffle_keyvalues(rnd, insert_kvs, kNumKeyValues / 2);

  // Begin test
  printf("Testing insertion/deletion of %d/%d elements into GPU hash table...\n",
      (uint32_t)insert_kvs.size(), (uint32_t)delete_kvs.size());

  Time timer = start_timer();

  // Create a hash table. For linear probing, this is just an array of KeyValues
  KeyValue* pHashTable;
  iris_alloc((void**) &pHashTable, sizeof(KeyValue) * kHashTableCapacity);

  // Initialize hash table to empty
  static_assert(kEmpty == 0xffffffff, "memset expected kEmpty=0xffffffff");
  iris_set(pHashTable, kEmpty, sizeof(KeyValue) * kHashTableCapacity);

  double total_ktime = 0.0;

  // Insert items into the hash table
  uint32_t num_inserts_per_batch = (uint32_t)insert_kvs.size() / num_insert_batches;

  // Create insert key values
  KeyValue* pInsertKvs;
  iris_alloc ((void**) &pInsertKvs, sizeof(KeyValue) * num_inserts_per_batch);

  for (uint32_t i = 0; i < num_insert_batches; i++)
  {
    iris_copy(pInsertKvs, insert_kvs.data() + i * num_inserts_per_batch,
               sizeof(KeyValue) * num_inserts_per_batch, IRISHostToDevice);

    total_ktime += insert_hashtable(pHashTable, pInsertKvs, num_inserts_per_batch);
  }
  printf("Average kernel execution time (insert): %f (s)\n", (total_ktime * 1e-9) / num_insert_batches);

  // Delete items from the hash table
  total_ktime = 0.0;
  uint32_t num_deletes_per_batch = (uint32_t)delete_kvs.size() / num_delete_batches;

  // Create delete key values
  KeyValue* pDeleteKvs;
  iris_alloc ((void**) &pDeleteKvs, sizeof(KeyValue) * num_deletes_per_batch);

  for (uint32_t i = 0; i < num_delete_batches; i++)
  {
    iris_copy(pDeleteKvs, delete_kvs.data() + i * num_deletes_per_batch,
               sizeof(KeyValue) * num_deletes_per_batch, IRISHostToDevice);

    total_ktime += delete_hashtable(pHashTable, pDeleteKvs, num_deletes_per_batch);
  }
  printf("Average kernel execution time (delete): %f (s)\n", (total_ktime * 1e-9) / num_delete_batches);

  // Get all the key-values from the hash table
  std::vector<KeyValue> kvs = iterate_hashtable(pHashTable);

  iris_free(pHashTable);
  iris_free(pInsertKvs);
  iris_free(pDeleteKvs);

  // Summarize results
  double milliseconds = get_elapsed_time(timer);
  double seconds = milliseconds / 1000.0f;
  printf("Total time (including memory copies, readback, etc): %f ms (%f million keys/second)\n", milliseconds,
      kNumKeyValues / seconds / 1000000.0f);

  test_unordered_map(insert_kvs, delete_kvs);

  bool ok = test_correctness(insert_kvs, delete_kvs, kvs);

  printf("%s\n", ok ? "PASS" : "FAIL");

  return 0;
}
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <chrono>
#include "linearprobing.h"

// 32 bit Murmur3 hash


// Insert the key/values in kvs into the hashtable


double insert_hashtable(KeyValue* pHashTable, const KeyValue* kvs, uint32_t num_kvs)
{
  // Insert all the keys into the hash table
  const int threadblocksize = 256;
  int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;

  iris_sync();
  auto start = std::chrono::steady_clock::now();

  iris_launch("k_hashtable_insert", gridsize, threadblocksize, pHashTable, kvs, (uint32_t)num_kvs);

  iris_sync();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  
  return time;
}

// Delete each key in kvs from the hash table, if the key exists
// A deleted key is left in the hash table, but its value is set to kEmpty
// Deleted keys are not reused; once a key is assigned a slot, it never moves


double delete_hashtable(KeyValue* pHashTable, const KeyValue* kvs, uint32_t num_kvs)
{
  // Insert all the keys into the hash table
  const int threadblocksize = 256;
  int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;

  iris_sync();
  auto start = std::chrono::steady_clock::now();

  iris_launch("k_hashtable_delete", gridsize, threadblocksize, pHashTable, kvs, (uint32_t)num_kvs);

  iris_sync();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return time;
}

// Iterate over every item in the hashtable; return non-empty key/values


std::vector<KeyValue> iterate_hashtable(KeyValue* pHashTable)
{
  uint32_t* device_num_kvs;
  iris_alloc((void**) &device_num_kvs, sizeof(uint32_t));
  iris_set(device_num_kvs, 0, sizeof(uint32_t));

  KeyValue* device_kvs;
  iris_alloc((void**) &device_kvs, sizeof(KeyValue) * kNumKeyValues);

  const int threadblocksize = 256;
  int gridsize = (kHashTableCapacity + threadblocksize - 1) / threadblocksize;

  iris_sync();
  auto start = std::chrono::steady_clock::now();

  iris_launch("k_iterate_hashtable", gridsize, threadblocksize, pHashTable, device_kvs, device_num_kvs);

  iris_sync();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Kernel execution time (iterate): %f (s)\n", time * 1e-9f);

  uint32_t num_kvs;
  iris_copy(&num_kvs, device_num_kvs, sizeof(uint32_t), IRISDeviceToHost);

  std::vector<KeyValue> kvs;
  kvs.resize(num_kvs);

  iris_copy(kvs.data(), device_kvs, sizeof(KeyValue) * num_kvs, IRISDeviceToHost);

  iris_free(device_kvs);
  iris_free(device_num_kvs);

  return kvs;
}
#include <stdio.h>
#include <stdint.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <random>
#include "linearprobing.h"

bool test_correctness(std::vector<KeyValue> insert_kvs, std::vector<KeyValue> delete_kvs, std::vector<KeyValue> kvs)
{
  printf("Testing that there are no duplicate keys...\n");
  std::unordered_set<uint32_t> unique_keys;
  for (uint32_t i = 0; i < kvs.size(); i++)
  {
    if (i % 10000000 == 0)
      printf("    Verifying %d/%d\n", i, (uint32_t)kvs.size());

    KeyValue* node = &kvs[i];
    if (unique_keys.find(node->key) != unique_keys.end())
    {
      printf("Duplicate key found in GPU hash table at slot %d\n", i);
      return false;
    }
    unique_keys.insert(node->key);
  }

  printf("Building unordered_map from original list...\n");
  std::unordered_map<uint32_t, std::vector<uint32_t>> all_kvs_map;
  for (uint32_t i = 0; i < (uint32_t)insert_kvs.size(); i++)
  {
    if (i % 10000000 == 0)
      printf("    Inserting %u/%u\n", i, (uint32_t)insert_kvs.size());

    auto iter = all_kvs_map.find(insert_kvs[i].key);
    if (iter == all_kvs_map.end())
    {
      all_kvs_map[insert_kvs[i].key] = std::vector<uint32_t>({ insert_kvs[i].value });
    }
    else
    {
      iter->second.push_back(insert_kvs[i].value);
    }
  }

  for (uint32_t i = 0; i < (uint32_t)delete_kvs.size(); i++)
  {
    if (i % 10000000 == 0)
      printf("    Deleting %u/%u\n", i, (uint32_t)delete_kvs.size());

    auto iter = all_kvs_map.find(delete_kvs[i].key);
    if (iter != all_kvs_map.end())
    {
      all_kvs_map.erase(iter);
    }
  }

  if (unique_keys.size() != all_kvs_map.size())
  {
    printf("# of unique keys in hashtable is incorrect\n");
    return false;
  }

  printf("Testing that each key/value in hashtable is in the original list...\n");
  for (uint32_t i = 0; i < kvs.size(); i++)
  {
    if (i % 10000000 == 0)
      printf("    Verifying %d/%d\n", i, (uint32_t)kvs.size());

    auto iter = all_kvs_map.find(kvs[i].key);
    if (iter == all_kvs_map.end())
    {
      printf("Hashtable key not found in original list\n");
      return false;
    }

    std::vector<uint32_t>& values = iter->second;
    if (std::find(values.begin(), values.end(), kvs[i].value) == values.end())
    {
      printf("Hashtable value not found in original list\n");
      return false;
    }
  }

  printf("Deleting std::unordered_map and std::unique_set...\n");

  return true;
}
