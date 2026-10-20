#include "study_check.h"
extern "C" int interval_check_coverage(const void*raw,const int*counts,int*missing,int*extra){auto c=check_coverage(raw,counts);*missing=c.missing;*extra=c.extra;return c.pass;}
