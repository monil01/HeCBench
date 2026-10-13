#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <utility>
#include "matrix.h"

template<class T> struct LanczosRecord {
  int steps;
  std::vector<T> alpha, beta, vectors, eigen;
};
template<class T> struct LanczosObserver {
  static std::vector<LanczosRecord<T>>& records() { static std::vector<LanczosRecord<T>> r; return r; }
  static void capture(int steps,const symm_tridiag_matrix<T>& tri,
                      const std::vector<T>& x,const std::vector<T>& prev,const std::vector<T>& y) {
    if (!getenv("LANCZOS_DUMP")) return;
    LanczosRecord<T> r; r.steps=steps;
    r.alpha.assign(tri.alpha_data(),tri.alpha_data()+steps);
    r.beta.assign(tri.beta_data(),tri.beta_data()+steps);
    if (getenv("LANCZOS_DUMP_VECTORS")) {
      r.vectors.insert(r.vectors.end(),x.begin(),x.end());
      r.vectors.insert(r.vectors.end(),prev.begin(),prev.end());
      r.vectors.insert(r.vectors.end(),y.begin(),y.end());
    }
    records().push_back(std::move(r));
  }
  static void write() {
    const char* prefix=getenv("LANCZOS_DUMP"); if(!prefix) return;
    std::string path=std::string(prefix)+".full";
    FILE* f=fopen(path.c_str(),"w"); if(!f) abort();
    for(const auto& r:records()) {
      fprintf(f,"RECORD %d %zu %zu %zu %zu\n",r.steps,sizeof(T),r.alpha.size(),r.vectors.size(),r.eigen.size());
      for(const auto* values:{&r.alpha,&r.beta,&r.vectors,&r.eigen})
        for(T v:*values) fprintf(f,"%a\n",double(v));
    }
    if(fclose(f)) abort(); records().clear();
  }
};
template<class T> void lanczos_eigen_observer(int steps,const std::vector<T>& eigen) {
  if(!getenv("LANCZOS_DUMP")) return;
  auto& records=LanczosObserver<T>::records();
  if(records.empty() || records.back().steps!=steps) abort();
  records.back().eigen=eigen;
}
