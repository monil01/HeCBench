#pragma once
#include "matrix.h"
#include <set>
#include <utility>
inline coo_matrix<double> control_graph(int caseid){
 const int sz[5][3]={{6,5,4},{17,13,9},{25,19,13},{37,29,23},{53,43,31}};
 const int n=257;coo_matrix<double> out(n);std::set<std::pair<int,int>> e;
 auto add=[&](int i,int j){if(i!=j){e.insert({i,j});e.insert({j,i});}};
 int start=0,prev=-1;
 for(int c=0;c<3;c++){int count=sz[caseid][c];for(int i=start;i<start+count;i++)for(int j=i+1;j<start+count;j++)add(i,j);if(prev>=0)add(prev,start);prev=start+count-1;start+=count;}
 add(prev,start);for(int i=start;i<n-1;i++)add(i,i+1);
 double one=1;for(auto edge:e)out.add_entry(edge.first,edge.second,one);return out;
}
