#pragma once
#include <vector>
#include <array>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <string>
struct SnapInput{
 uint32_t h[12];double scalars[4];int J,na,ng,nb,ncoeff,nu,nd,nz,ncg,jdim,pdim,nt;
 std::vector<int32_t>iu,parity,id,iz,icg,inside;std::vector<double>root,cg,zbeta,rij,w,rcut,reference,coeff;
 template<class T>void rd(std::ifstream&f,T*p,size_t n){f.read((char*)p,sizeof(T)*n);if(!f)throw std::runtime_error("fixture short read");}
 template<class T>void rd(std::ifstream&f,std::vector<T>&v,size_t n){v.resize(n);rd(f,v.data(),n);}
 explicit SnapInput(const std::string&path){std::ifstream f(path,std::ios::binary);rd(f,h,12);if(h[0]!=0x534e4150u)throw std::runtime_error("fixture magic");J=h[1];na=h[2];ng=h[3];nb=h[4];ncoeff=h[5];nu=h[6];nd=h[7];nz=h[8];ncg=h[9];jdim=h[10];pdim=h[11];nt=na+ng;rd(f,scalars,4);rd(f,iu,jdim);rd(f,parity,nu);rd(f,id,jdim);rd(f,iz,9*nz);rd(f,icg,jdim*jdim*jdim);rd(f,root,pdim*pdim);rd(f,cg,ncg);rd(f,zbeta,nz);rd(f,rij,na*nb*3);rd(f,inside,na*nb);rd(f,w,na*nb);rd(f,rcut,na*nb);rd(f,reference,nt*3);rd(f,coeff,ncoeff+1);char b;if(f.read(&b,1))throw std::runtime_error("fixture trailing data");}
};
