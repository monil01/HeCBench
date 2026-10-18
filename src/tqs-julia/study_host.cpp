#include "../tqs-cuda/host_task.cpp"
extern "C" void study_insert(task_t*q,int*d,task_t*pool,int*data,int*written,int Q,int offset,int T){host_insert_tasks(q,d,pool,data,written,Q,offset,T);}
