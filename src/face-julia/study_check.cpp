#include "study_support.h"
static std::vector<FCase>cs=face_load();
extern "C" int face_ncases(){return cs.size();}
extern "C" int face_dim(int k,int d){return d?cs.at(k).h:cs.at(k).w;}
extern "C" int face_ninput(int k){return cs.at(k).input.size();}
extern "C" const int32_t* face_input(int k){return cs.at(k).input.data();}
extern "C" int face_validate(const int64_t*p,int64_t n){return face_check(p,n);}
