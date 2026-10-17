#include <cstdio>
#include <cstdint>
int main(){const int n=1;double a[2]={-777.,-888.};long long b[2]={-777,-888};double*input1=a;long long*input2=b;for(int i=1;i<=n;i++){input1[i]=22.44/i;input2[i]=0x403670A3D70A3D71;}printf("n=1 logical_input0=%.1f/int%lld guard_slot1=%.17g/int%llx; original loop leaveslogical0unchangedandwriteslogicaln\n",a[0],b[0],a[1],b[1]);return !(a[0]==-777.&&b[0]==-777&&a[1]==22.44&&b[1]==0x403670A3D70A3D71);}
