#include <iris/iris.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#define BufSize 0x1000
#define Iterations 0x100
#define TotalStreams 4
#define TotalBufs 4

class PerfStreamCreateCopyDestroy {
  private:
    unsigned int numBuffers_;
    unsigned int numStreams_;
    const size_t totalStreams_[TotalStreams];
    const size_t totalBuffers_[TotalBufs];

  public:
    PerfStreamCreateCopyDestroy() : numBuffers_(0), numStreams_(0),
                                       totalStreams_{1, 2, 4, 8},
                                       totalBuffers_{1, 100, 1000, 5000} {};
    ~PerfStreamCreateCopyDestroy() {};
    void run_baseline(unsigned int testNumber);
    void run_stream(unsigned int testNumber);
};

static void fill_src(std::vector<float>& hSrc) {
  for (size_t i = 0; i < BufSize; i++) {
    hSrc[i] = 1.618f + i;
  }
}

void PerfStreamCreateCopyDestroy::run_stream(unsigned int testNumber) {
  numStreams_ = totalStreams_[testNumber % TotalStreams];
  size_t iter = Iterations / (numStreams_ * ((size_t)1 << (testNumber / TotalBufs + 1)));

  numBuffers_ = totalBuffers_[testNumber / TotalBufs];
  size_t nBytes = BufSize * sizeof(float);

  std::vector<float> hSrc(BufSize);
  fill_src(hSrc);

  std::vector<std::vector<float> > buffers(numBuffers_, std::vector<float>(BufSize));
  std::vector<std::unique_ptr<iris::DMem> > dSrc;
  dSrc.reserve(numBuffers_);
  for (size_t b = 0; b < numBuffers_; ++b) {
    dSrc.emplace_back(new iris::DMem(buffers[b].data(), nBytes));
  }

  auto start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < iter; ++i) {
    for (size_t s = 0; s < numStreams_; ++s) {
      for (size_t b = 0; b < numBuffers_; ++b) {
        dSrc[b]->update(hSrc.data());
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff = end - start;

  auto time = static_cast<float>(diff.count() * 1000 / (iter * numStreams_));

  std::cout << "[Stream] Create+Copy+Synchronize+Destroy time for " << numStreams_ << " streams and "
       << std::setw(4) << numBuffers_ << " buffers " << " and " << std::setw(4)
       << iter << " iterations " << time << " (ms) " << std::endl;
}

void PerfStreamCreateCopyDestroy::run_baseline(unsigned int testNumber) {
  numStreams_ = totalStreams_[testNumber % TotalStreams];
  size_t iter = Iterations / (numStreams_ * ((size_t)1 << (testNumber / TotalBufs + 1)));

  numBuffers_ = totalBuffers_[testNumber / TotalBufs];
  size_t nBytes = BufSize * sizeof(float);

  std::vector<float> hSrc(BufSize);
  fill_src(hSrc);

  std::vector<std::vector<float> > buffers(numBuffers_, std::vector<float>(BufSize));
  std::vector<std::unique_ptr<iris::DMem> > dSrc;
  dSrc.reserve(numBuffers_);
  for (size_t b = 0; b < numBuffers_; ++b) {
    dSrc.emplace_back(new iris::DMem(buffers[b].data(), nBytes));
  }

  auto start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < iter; ++i) {
    for (size_t s = 0; s < numStreams_; ++s) {
      for (size_t b = 0; b < numBuffers_; ++b) {
        dSrc[b]->update(hSrc.data());
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff = end - start;

  auto time = static_cast<float>(diff.count() * 1000 / (iter * numStreams_));

  std::cout << "[Baseline] Copy+Synchronize time for the default stream and "
       << std::setw(4) << numBuffers_ << " buffers " << " and " << std::setw(4)
       << iter << " iterations " << time << " (ms) " << std::endl;
}

int main(int argc, char* argv[]) {
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  PerfStreamCreateCopyDestroy streamCCD;

  streamCCD.run_baseline(0);
  for (auto testCase = 0; testCase < TotalStreams * TotalBufs; testCase++) {
    streamCCD.run_baseline(testCase);
  }

  streamCCD.run_stream(0);
  for (auto testCase = 0; testCase < TotalStreams * TotalBufs; testCase++) {
    streamCCD.run_stream(testCase);
  }

  platform.finalize();
}
