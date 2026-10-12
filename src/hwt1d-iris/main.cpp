#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

template <typename T>
static T roundToPowerOf2(T val)
{
  int bytes = sizeof(T);
  val--;
  for (int i = 0; i < bytes; i++) {
    val |= val >> (1 << i);
  }
  val++;
  return val;
}

static int getLevels(unsigned int length, unsigned int* levels)
{
  for (unsigned int i = 0; i < 24; ++i) {
    if (length == (1U << i)) {
      *levels = i;
      return 0;
    }
  }
  return 1;
}

static void calApproxFinalOnHost(float* inData, float* hOutData, unsigned int signalLength)
{
  std::vector<float> tempOutData(signalLength);
  std::memcpy(tempOutData.data(), inData, signalLength * sizeof(float));

  for (unsigned int i = 0; i < signalLength; ++i) {
    tempOutData[i] = tempOutData[i] / std::sqrt((float)signalLength);
  }

  unsigned int length = signalLength;
  while (length > 1u) {
    for (unsigned int i = 0; i < length / 2; ++i) {
      float data0 = tempOutData[2 * i];
      float data1 = tempOutData[2 * i + 1];
      hOutData[i] = (data0 + data1) / std::sqrt(2.0f);
      hOutData[length / 2 + i] = (data0 - data1) / std::sqrt(2.0f);
    }
    std::memcpy(tempOutData.data(), hOutData, signalLength * sizeof(float));
    length >>= 1;
  }
}

static int runKernel(
    float* inData,
    float* dOutData,
    float* hOutData,
    float* dPartialOutData,
    unsigned int signalLength,
    iris::DMem& inDataMem,
    iris::DMem& partialOutDataMem,
    iris::DMem& outDataMem)
{
  unsigned int levels = 0;
  if (getLevels(signalLength, &levels) == 1) {
    std::cerr << "getLevels() failed\n";
    return 1;
  }

  unsigned int actualLevels = levels;
  unsigned int maxLevelsOnDevice = 9;
  const unsigned int signalLengthByte = signalLength * sizeof(float);
  std::vector<float> temp(inData, inData + signalLength);

  unsigned int levelsDone = 0;
  const unsigned int one = 1;
  while (levelsDone < actualLevels) {
    unsigned int curLevels = (levels < maxLevelsOnDevice) ? levels : maxLevelsOnDevice;
    unsigned int curSignalLength = (levelsDone == 0) ? signalLength : (one << levels);
    unsigned int groupSize = (one << curLevels) / 2;
    unsigned int totalLevels = levels;
    unsigned int grid = (curSignalLength >> 1) / groupSize;

    inDataMem.update(inData);

    size_t gws[1] = {static_cast<size_t>(grid) * groupSize};
    size_t lws[1] = {groupSize};
    iris::Task task;
    void* params[7] = {
      &inDataMem,
      &outDataMem,
      &partialOutDataMem,
      &totalLevels,
      &curSignalLength,
      &levelsDone,
      &maxLevelsOnDevice
    };
    int pinfo[7] = {
      iris_r,
      iris_w,
      iris_w,
      sizeof(totalLevels),
      sizeof(curSignalLength),
      sizeof(levelsDone),
      sizeof(maxLevelsOnDevice)
    };
    task.kernel("dwtHaar1D", 1, nullptr, gws, lws, 7, params, pinfo);
    task.flush_out(outDataMem);
    task.flush_out(partialOutDataMem);
    task.submit(iris_gpu, nullptr, 1);

    if (levels <= maxLevelsOnDevice) {
      dOutData[0] = dPartialOutData[0];
      std::memcpy(hOutData, dOutData, (one << curLevels) * sizeof(float));
      std::memcpy(dOutData + (one << curLevels), hOutData + (one << curLevels),
                  (signalLength - (one << curLevels)) * sizeof(float));
      break;
    }

    levels -= maxLevelsOnDevice;
    std::memcpy(hOutData, dOutData, curSignalLength * sizeof(float));
    std::memcpy(inData, dPartialOutData, (one << levels) * sizeof(float));
    levelsDone += maxLevelsOnDevice;
  }

  std::memcpy(inData, temp.data(), signalLengthByte);
  return 0;
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <signal length> <repeat>\n";
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  unsigned int signalLength = std::atoi(argv[1]);
  const int iterations = std::atoi(argv[2]);
  signalLength = roundToPowerOf2<unsigned int>(signalLength);

  unsigned int levels = 0;
  if (getLevels(signalLength, &levels) == 1) {
    std::cerr << "signalLength > 2 ^ 23 not supported\n";
    platform.finalize();
    return 1;
  }

  std::vector<float> inData(signalLength);
  std::srand(2);
  for (unsigned int i = 0; i < signalLength; i++) {
    inData[i] = (float)(std::rand() % 10);
  }

  std::vector<float> dOutData(signalLength, 0.0f);
  std::vector<float> dPartialOutData(signalLength, 0.0f);
  std::vector<float> hOutData(signalLength, 0.0f);

  iris::DMem inDataMem(inData.data(), signalLength * sizeof(float));
  iris::DMem partialOutDataMem(dPartialOutData.data(), signalLength * sizeof(float));
  iris::DMem outDataMem(dOutData.data(), signalLength * sizeof(float));

  std::cout << "Executing kernel for " << iterations << " iterations" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; i++) {
    if (runKernel(inData.data(), dOutData.data(), hOutData.data(), dPartialOutData.data(),
                  signalLength, inDataMem, partialOutDataMem, outDataMem) != 0) {
      platform.finalize();
      return 1;
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average device offload time " << (time * 1e-9f) / iterations << " (s)\n";

  std::fill(hOutData.begin(), hOutData.end(), 0.0f);
  calApproxFinalOnHost(inData.data(), hOutData.data(), signalLength);

  bool ok = true;
  for (unsigned int i = 0; i < signalLength; ++i) {
    if (std::fabs(dOutData[i] - hOutData[i]) > 0.1f) {
      ok = false;
      break;
    }
  }

  platform.finalize();
  std::cout << (ok ? "PASS" : "FAIL") << std::endl;
  return 0;
}
