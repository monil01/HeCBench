#pragma once
inline void export_verification(const char *suffix,const int *data,int n) {
 if(const char *path=getenv("HECBENCH_VALIDATION_DUMP")) {
   char name[4096];snprintf(name,sizeof(name),"%s%s",path,suffix);
   FILE *f=fopen(name,"wb");if(!f)exit(4);
   if(fwrite(data,sizeof(int),n,f)!=size_t(n))exit(4);fclose(f);
 }
}
