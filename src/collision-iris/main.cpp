#include <iris/iris.hpp>

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <vector>

using namespace std;

static vector<int> checkDuplicates(const vector<int>& v)
{
  vector<int> hasDuplicates(32, 0);
  iris::DMem devSet(const_cast<int*>(v.data()), v.size() * sizeof(int));
  iris::DMem devOut(hasDuplicates.data(), hasDuplicates.size() * sizeof(int));

  size_t gws[1] = {32};
  size_t lws[1] = {32};
  int num = static_cast<int>(v.size());
  iris::Task task;
  void* params[3] = {&num, &devSet, &devOut};
  int pinfo[3] = {sizeof(num), iris_r, iris_w};
  task.kernel("checkDuplicates", 1, nullptr, gws, lws, 3, params, pinfo);
  task.flush_out(devOut);
  task.submit(iris_gpu, nullptr, 1);
  devOut.fetch(hasDuplicates.data());
  return hasDuplicates;
}

static unsigned int checkDuplicateMask(const vector<int>& v)
{
  unsigned int mask = 0;
  iris::DMem devSet(const_cast<int*>(v.data()), v.size() * sizeof(int));
  iris::DMem devMask(&mask, sizeof(mask));

  size_t gws[1] = {32};
  size_t lws[1] = {32};
  int num = static_cast<int>(v.size());
  iris::Task task;
  void* params[3] = {&num, &devSet, &devMask};
  int pinfo[3] = {sizeof(num), iris_r, iris_w};
  task.kernel("checkDuplicateMask", 1, nullptr, gws, lws, 3, params, pinfo);
  task.flush_out(devMask);
  task.submit(iris_gpu, nullptr, 1);
  devMask.fetch(&mask);
  return mask;
}

static void test_collision(const int ND)
{
  for (int numDups = 0; numDups < ND; ++numDups) {
    vector<int> v;
    for (int i = 0; i < ND - numDups; ++i) {
      int r = 0;
      while (true) {
        r = rand();
        bool found = false;
        for (unsigned int j = 0; j < v.size(); ++j) {
          if (v[j] == r) {
            found = true;
            break;
          }
        }
        if (!found) break;
      }
      v.push_back(r);
    }
    for (int i = 0; i < numDups; ++i) v.push_back(v[0]);
    assert(ND == static_cast<int>(v.size()));
    auto dupCheck = checkDuplicates(v);
    for (auto dup : dupCheck) assert((numDups > 0) == static_cast<bool>(dup));
  }
}

static void test_collisionMask(const int ND)
{
  for (int numDups = 0; numDups < ND; ++numDups) {
    vector<int> v;
    for (int i = 0; i < ND - numDups; ++i) {
      int r = 0;
      while (true) {
        r = rand();
        bool found = false;
        for (unsigned int j = 0; j < v.size(); ++j) {
          if (v[j] == r) {
            found = true;
            break;
          }
        }
        if (!found) break;
      }
      v.push_back(r);
    }
    for (int i = 0; i < numDups; ++i) v.push_back(v[0]);
    assert(ND == static_cast<int>(v.size()));
    auto mask = checkDuplicateMask(v);
    auto expected = numDups > 0 ? 0xffffffffU << (ND - numDups) : 0;
    if (expected != mask) {
      printf("Error: numDups=%d expected=%x mask=%x\n", numDups, expected, mask);
      break;
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  srand(123);
  const int num_dup = 32;
  const int repeat = atoi(argv[1]);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) test_collision(num_dup);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the function test_collision: %f (us)\n",
         time * 1e-3f / repeat);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) test_collisionMask(num_dup);
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the function test_collisionMask: %f (us)\n",
         time * 1e-3f / repeat);

  platform.finalize();
  return 0;
}
