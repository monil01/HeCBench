// Original serial FRNA path; ASan witness of shifted w5 initialization.
#define INT
#include "../frna-cuda/frna.cu"
int main(int argc,char**argv){if(argc!=2)return 2;fparam par;fparam_read_from_text(argv[1],0,&par);frna_t p=frna_new("GCGAAACGC",&par);frna_delete(p);}
