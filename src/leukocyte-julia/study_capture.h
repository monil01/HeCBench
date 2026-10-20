#pragma once
#include <cuda_runtime.h>
#include <unistd.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace leukostudy {
constexpr size_t guard_bytes = 256;
inline void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
struct Allocation { void* base; void* data; size_t bytes; std::string name; };
struct State {
  std::map<std::string,Allocation> live;
  std::vector<double> prefix, output;
  double seconds = 0;
  bool capture = false, correct = true;
  int trial = 0, frames = 0, frame = 0, cells = 0;
  std::string folder, path;
  std::ofstream raw, guards, layout, timings;
  std::vector<int> iterations;
};
inline State& state() { static State value; return value; }
inline std::string allocation_name(const char* expression) {
  std::string name(expression); auto start=name.find("d_");
  if(start==std::string::npos) throw std::runtime_error("unrecognized GPU allocation name");
  size_t end=start;
  while(end<name.size() && (std::isalnum((unsigned char)name[end]) || name[end]=='_')) end++;
  return name.substr(start,end-start);
}
inline bool integer_buffer(const std::string& name) {
  return name=="d_tX" || name=="d_tY" || name=="d_I_offsets" ||
         name=="d_m_array" || name=="d_n_array" || name=="d_iterations";
}
inline void write_bytes(std::ofstream& output,const void* data,size_t bytes) {
  output.write((const char*)data,bytes);
  if(!output) throw std::runtime_error("retained evidence write failed");
}
inline cudaError_t allocate(void** pointer,size_t bytes,const char* expression) {
  auto& s=state(); std::string name=allocation_name(expression);
  if(s.live.count(name)) throw std::runtime_error("GPU allocation lifetime overlaps: "+name);
  void* base=nullptr; check(cudaMalloc(&base,bytes+2*guard_bytes));
  check(cudaMemset(base,165,bytes+2*guard_bytes));
  *pointer=(unsigned char*)base+guard_bytes;
  s.live.emplace(name,Allocation{base,*pointer,bytes,name});
  return cudaSuccess;
}
inline void retain_guards(const unsigned char* before,const unsigned char* after) {
  auto& s=state();
  bool valid=std::all_of(before,before+guard_bytes,[](auto b){return b==165;}) &&
             std::all_of(after,after+guard_bytes,[](auto b){return b==165;});
  s.correct &= valid;
  if(s.capture) {
    write_bytes(s.guards,before,guard_bytes);
    write_bytes(s.guards,after,guard_bytes);
  }
}
inline cudaError_t release(void* pointer) {
  auto& s=state(); auto found=s.live.end();
  for(auto it=s.live.begin();it!=s.live.end();++it) if(it->second.data==pointer) {found=it;break;}
  if(found==s.live.end()) throw std::runtime_error("release of unknown GPU allocation");
  auto allocation=found->second;
  unsigned char first[guard_bytes],last[guard_bytes];
  check(cudaMemcpy(first,allocation.base,guard_bytes,cudaMemcpyDeviceToHost));
  check(cudaMemcpy(last,(unsigned char*)allocation.data+allocation.bytes,guard_bytes,cudaMemcpyDeviceToHost));
  if(s.capture) s.layout << "{\"kind\":\"release_guard\",\"name\":\"" << allocation.name
      << "\",\"frame\":" << s.frame << ",\"guard_offset_bytes\":" << s.guards.tellp()
      << ",\"guard_bytes\":512}\n";
  retain_guards(first,last);
  check(cudaFree(allocation.base)); s.live.erase(found);
  return cudaSuccess;
}
inline void append_values(std::vector<double>& destination,const std::vector<unsigned char>& bytes,bool integer) {
  if(bytes.size()%4) throw std::runtime_error("non-four-byte GPU field");
  for(size_t at=0;at<bytes.size();at+=4) {
    double value;
    if(integer) { int x; std::memcpy(&x,bytes.data()+at,4); value=x; }
    else { float x; std::memcpy(&x,bytes.data()+at,4); value=x; }
    if(!std::isfinite(value)) state().correct=false;
    destination.push_back(value);
  }
}
inline std::map<std::string,std::vector<unsigned char>> capture_gpu(const char* stage,const char* phase) {
  auto& s=state(); std::map<std::string,std::vector<unsigned char>> logical;
  if(!s.capture) return logical;
  s.layout << "{\"kind\":\"gpu_buffers\",\"stage\":\"" << stage
      << "\",\"phase\":\"" << phase << "\",\"frame\":" << s.frame << ",\"buffers\":[";
  bool first=true;
  for(const auto& item:s.live) {
    const auto& a=item.second;
    std::vector<unsigned char> physical(a.bytes+2*guard_bytes);
    check(cudaMemcpy(physical.data(),a.base,physical.size(),cudaMemcpyDeviceToHost));
    logical.emplace(a.name,std::vector<unsigned char>(physical.begin()+guard_bytes,physical.end()-guard_bytes));
    if(!first) s.layout << ','; first=false;
    s.layout << "{\"name\":\"" << a.name << "\",\"dtype\":\""
        << (integer_buffer(a.name)?"int32":"float32") << "\",\"raw_offset_bytes\":" << s.raw.tellp()
        << ",\"physical_bytes\":" << physical.size() << ",\"payload_offset_bytes\":256,\"payload_bytes\":" << a.bytes
        << ",\"guard_offset_bytes\":" << s.guards.tellp() << ",\"guard_bytes\":512}";
    write_bytes(s.raw,physical.data(),physical.size());
    retain_guards(physical.data(),physical.data()+guard_bytes+a.bytes);
  }
  s.layout << "]}\n";
  return logical;
}
inline void frame_bytes(int frame,int width,int height,int cropped,int scaled,
                        const unsigned char* storage,const unsigned char* image) {
  auto& s=state(); s.frame=frame;
  bool valid=std::all_of(storage,storage+32,[](auto x){return x==165;}) &&
      std::all_of(image+size_t(width)*height,image+size_t(width)*height+32,[](auto x){return x==165;});
  s.correct &= valid;
  if(!s.capture) return;
  size_t bytes=size_t(width)*height;
  s.layout << "{\"kind\":\"video_frame\",\"frame\":" << frame
      << ",\"width\":" << width << ",\"height\":" << height
      << ",\"cropped\":" << cropped << ",\"scaled\":" << scaled
      << ",\"raw_offset_bytes\":" << s.raw.tellp() << ",\"payload_bytes\":" << bytes
      << ",\"guard_offset_bytes\":" << s.guards.tellp() << ",\"guard_bytes\":64}\n";
  write_bytes(s.raw,image,bytes);
  write_bytes(s.guards,storage,32); write_bytes(s.guards,image+bytes,32);
  s.prefix.insert(s.prefix.end(),{double(frame),double(width),double(height),double(cropped),double(scaled)});
  for(size_t start=0;start<bytes;start+=6) {
    uint64_t word=0;
    for(size_t j=0;j<6 && start+j<bytes;j++) word |= uint64_t(image[start+j])<<(8*j);
    s.prefix.push_back(double(word));
  }
}
inline void before_gicov(int m,int n,int local,int groups,int threads,int blocks) {
  auto& s=state(); auto values=capture_gpu("GICOV","before");
  if(!s.capture) return;
  s.prefix.insert(s.prefix.end(),{double(m),double(n),double(local),double(groups),double(threads),double(blocks)});
  for(const char* name:{"d_cos_angle","d_grad_x","d_grad_y","d_sin_angle","d_tX","d_tY"})
    append_values(s.prefix,values.at(name),integer_buffer(name));
}
inline void before_dilation(int m,int n,int sm,int sn,int threads,int blocks) {
  auto& s=state(); auto values=capture_gpu("DILATION","before");
  if(!s.capture) return;
  s.prefix.insert(s.prefix.end(),{double(sm),double(sn),double(m),double(n),double(threads),double(blocks)});
  append_values(s.prefix,values.at("d_strel"),false);
  s.output.insert(s.output.end(),{2.,double(s.frame),double(m),double(n)});
  append_values(s.output,values.at("d_gicov"),false);
}
inline void before_mgvf(int cells,float vx,float vy,float epsilon,float cutoff,int maximum) {
  auto& s=state(); s.cells=cells; auto values=capture_gpu("MGVF","before");
  if(!s.capture) return;
  s.prefix.insert(s.prefix.end(),{double(vx),double(vy),double(epsilon),double(cutoff),double(maximum)});
  s.output.insert(s.output.end(),{3.,double(s.frame),double(cells)});
  for(const char* name:{"d_I_offsets","d_m_array","d_n_array"}) append_values(s.output,values.at(name),true);
  if(values.at("d_I_all")!=values.at("d_IMGVF_all")) s.correct=false;
  append_values(s.output,values.at("d_I_all"),false);
}
inline void after(const char* stage,long long elapsed_nanoseconds) {
  auto& s=state(); double seconds=elapsed_nanoseconds*1e-9; s.seconds+=seconds;
  auto values=capture_gpu(stage,"after");
  if(!s.capture) return;
  if(std::string(stage)=="GICOV") {
    s.output.insert(s.output.end(),{1.,double(s.frame)});
    append_values(s.output,values.at("d_gicov"),false);
  } else if(std::string(stage)=="DILATION") {
    s.output.insert(s.output.end(),{2.,double(s.frame)});
    append_values(s.output,values.at("d_img_dilated"),false);
  } else {
    s.output.insert(s.output.end(),{3.,double(s.frame),double(s.cells)});
    append_values(s.output,values.at("d_IMGVF_all"),false);
    const auto& counts=values.at("d_iterations");
    s.iterations.resize(counts.size()/4); std::memcpy(s.iterations.data(),counts.data(),counts.size());
    for(int count:s.iterations) if(count<1 || count>500) s.correct=false;
  }
  s.timings << "{\"stage\":\"" << stage << "\",\"frame\":" << s.frame
      << ",\"seconds\":" << seconds << ",\"iterations\":[";
  if(std::string(stage)=="MGVF") for(size_t i=0;i<s.iterations.size();i++) {if(i)s.timings<<',';s.timings<<s.iterations[i];}
  s.timings << "]}\n";
}
inline void detection(int cells,const double* x,const double* y) {
  auto& s=state(); if(!s.capture) return;
  s.output.insert(s.output.end(),{4.,double(cells)});
  for(int c=0;c<cells;c++) {s.output.push_back(x[c]);s.output.push_back(y[c]);}
}
inline void trajectory(int frame,int cells,int snaxels,const double* x,const double* y,double** radii) {
  auto& s=state(); if(!s.capture) return;
  s.output.insert(s.output.end(),{5.,double(frame),double(cells),double(snaxels)});
  for(int c=0;c<cells;c++) {
    s.output.push_back(x[c]); s.output.push_back(y[c]);
    s.output.insert(s.output.end(),radii[c],radii[c]+snaxels);
  }
}
inline void begin_case(const std::string& folder,const std::string& model,int trial,int frames,bool capture) {
  auto& s=state(); if(!s.live.empty()) throw std::runtime_error("GPU buffers survived previous application case");
  s.capture=capture; s.frames=frames; s.frame=0; s.trial=trial; s.folder=folder;
  if(!capture) return;
  s.prefix.push_back(frames);
  s.path=folder+"/"+model+"_"+std::to_string(getpid())+"_trial"+std::to_string(trial)+"_case"+std::to_string(frames);
  s.raw.open(s.path+".raw.bin",std::ios::binary);
  s.guards.open(s.path+".guards.u8",std::ios::binary);
  s.layout.open(s.path+".layout.jsonl");
  s.timings.open(s.path+".timings.jsonl"); s.timings.precision(17);
  if(!s.raw || !s.guards || !s.layout || !s.timings) throw std::runtime_error("cannot open persistent case evidence");
}
inline void end_case() {
  auto& s=state(); if(!s.live.empty()) throw std::runtime_error("application failed to release GPU buffers");
  if(s.capture) {s.raw.close();s.guards.close();s.layout.close();s.timings.close();}
}
}
#define cudaMalloc(pointer,bytes) leukostudy::allocate((void**)(pointer),(bytes),#pointer)
#define cudaFree(pointer) leukostudy::release((void*)(pointer))
