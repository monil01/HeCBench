#include "study_support.h"
static std::vector<GCase>cs=load_cases();
extern "C" unsigned gerbil_ncases(){return cs.size();}
extern "C" unsigned gerbil_field(unsigned i,unsigned f){auto&c=cs.at(i);switch(f){case 0:return c.k;case 1:return c.ip;case 2:return c.n;case 3:return c.entries;case 4:return c.threshold;case 5:return c.unique;case 6:return c.expected.size();default:return 0;}}
extern "C" const uint32_t*gerbil_keys(unsigned i){return cs.at(i).keys.data();}
extern "C" int gerbil_validate(const uint64_t*p,uint64_t n){return check_record(p,n);}
