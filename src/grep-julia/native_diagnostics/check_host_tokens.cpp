#include "../../../src/grep-cuda/regex.h"
int main(int argc,char**argv){char*re=strdup(argv[1]);SimpleReBuilder b;simplifyRe(&re,&b);fwrite(b.re,1,strlen(b.re)+1,stdout);free(re);_simpleReBuilder(&b);return 0;}
