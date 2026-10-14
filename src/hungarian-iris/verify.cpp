#include "state.h"
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <limits>

// Independent shortest augmenting path solver, used only for verification.
// The GPU benchmark uses zero starring/priming and cover/slack updates.
static long long optimum(const std::vector<cost_t>& a) {
  std::vector<long long> u(n+1), v(n+1);
  std::vector<int> p(n+1), way(n+1);
  for (int row=1; row<=n; ++row) {
    p[0]=row;
    int col=0;
    std::vector<long long> best(n+1, std::numeric_limits<long long>::max()/4);
    std::vector<bool> used(n+1);
    do {
      used[col]=true;
      int r=p[col], next=0;
      long long delta=std::numeric_limits<long long>::max()/4;
      for (int c=1; c<=n; ++c) if (!used[c]) {
        long long reduced=static_cast<long long>(a[(c-1)*n+r-1])-u[r]-v[c];
        if (reduced<best[c]) { best[c]=reduced; way[c]=col; }
        if (best[c]<delta) { delta=best[c]; next=c; }
      }
      for (int c=0; c<=n; ++c) {
        if (used[c]) { u[p[c]]+=delta; v[c]-=delta; }
        else best[c]-=delta;
      }
      col=next;
    } while (p[col]);
    do { int prev=way[col]; p[col]=p[prev]; col=prev; } while (col);
  }
  return -v[0];
}

static std::vector<long long> costs(const char* path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error(std::string("Cannot read ")+path);
  std::vector<long long> out;
  std::string line;
  while (std::getline(in,line)) {
    auto pos=line.find("Total cost is");
    if (pos!=std::string::npos) out.push_back(std::stoll(line.substr(pos+13)));
  }
  return out;
}

int main(int argc, char** argv) {
  if (argc!=3) { std::cerr<<"Usage: verify native.txt iris.txt\n"; return 2; }
  try {
    auto native=costs(argv[1]), iris=costs(argv[2]);
#ifndef USE_TEST_MATRIX
    const int tests=n_tests;
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(0,range-1);
#else
    const int tests=1;
#endif
    if (native.size()!=static_cast<size_t>(tests) || iris.size()!=native.size())
      throw std::runtime_error("Missing numeric results");
    for (int t=0; t<tests; ++t) {
      std::vector<cost_t> a(n*n);
      for (int c=0; c<n; ++c) for (int r=0; r<n; ++r) {
#ifndef USE_TEST_MATRIX
        a[c*n+r]=(c<user_n && r<user_n) ? distribution(generator) : (c==r ? 0 : MAX_DATA);
#else
        a[c*n+r]=(c+1)*(r+1);
#endif
      }
      long long expected=optimum(a);
      if (native[t]!=expected || iris[t]!=expected) {
        std::cerr<<"FAIL test="<<t<<" native="<<native[t]<<" iris="<<iris[t]<<" optimum="<<expected<<'\n';
        return 1;
      }
    }
    std::cout<<"PASS n="<<n<<" tests="<<tests<<" exact native/IRIS/independent optimal costs\n";
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
