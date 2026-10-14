#pragma once
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <thread>
#include <stdexcept>
struct GpuLock {
 int fd;
 GpuLock(){
  fd=open("/tmp/hecbench-iris-validation.lock",O_CREAT|O_RDWR,0666);
  auto start=std::chrono::steady_clock::now();
  while(fd>=0 && flock(fd,LOCK_EX|LOCK_NB)!=0){
   if(std::chrono::steady_clock::now()-start>std::chrono::seconds(600)){close(fd);throw std::runtime_error("GPU lock timeout");}
   std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if(fd<0)throw std::runtime_error("GPU lock open failed");
 }
 ~GpuLock(){flock(fd,LOCK_UN);close(fd);}
};
