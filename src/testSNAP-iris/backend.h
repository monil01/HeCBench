#pragma once
inline void prove_backend() {
 int n=0; iris_device_count(&n); bool found=false;
 for(int i=0;i<n;++i) { int b=0,t=0; char name[256]={}; size_t z;
 iris_device_info(i,iris_backend,&b,&z); iris_device_info(i,iris_type,&t,&z); iris_device_info(i,iris_name,name,&z);
 printf("IRIS backend%d/type%d %s\n",b,t,name); if(b==3&&t==256) found=true; }
 if(!found) exit(2);
}
