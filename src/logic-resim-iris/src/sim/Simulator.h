#ifndef SIMULATOR_H
#define SIMULATOR_H
#include "layout.h"
#include <iris/iris.hpp>
#include <unordered_map>
#include <vector>
#include <memory>
class Simulator {
  tUnit dumpOff;
  size_t parallelSize=0;
  std::vector<char> eval;
  std::vector<gpu_GATE> gates;
  std::unordered_map<size_t, std::vector<Event>> histories;
public:
  explicit Simulator(tUnit off):dumpOff(off) {}
  void addEvalMgr(char*,size_t);
  void addTrans(tHistory*,size_t&);
  Event *getTrans(size_t&);
  void popTrans(size_t&);
  void addGateMgr(gpu_GATE*,size_t&);
  void simulateBlock(tHistory**,size_t*,size_t[][3]);
  void cleanGPU() { histories.clear(); gates.clear(); eval.clear(); }
  void simulate(Gate*);
  void simulateBlock(vector<Gate*>&);
};
#endif
