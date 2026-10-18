#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include "SNICIT_public.hpp"
int main(){auto*p=new SNICIT_BEY::SNICIT("unused","unused",128,18,.6f,128,8,1,1,false);p->_input_alloc_read("/home/imo/HeCBench_julia/src/snicit-cuda/dataset/beyond/MNIST/");return 0;}
