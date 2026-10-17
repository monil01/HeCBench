#pragma once
// Reconstructed missing host checker for original addConstant equation.
// The study harness also independently checks every output, so no PASS stub.
inline bool correctResult(const int* a,unsigned int n,int b,int repeat){int delta=0;for(int j=0;j<repeat;j++)delta+=j%b;for(unsigned int i=0;i<n;i++)if(a[i]!=int(i)+delta)return false;return true;}
