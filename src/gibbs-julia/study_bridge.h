extern "C" int study_state_bytes();
extern "C" void study_reset();
extern "C" int study_seed_states(void*,int);
extern "C" void study_feedback(float*,float*,int,float,float);
extern "C" int study_check(const void*,const void*,const float*,const int*,const float*,float,float,int,double*);
extern "C" int study_native_launch(void*,float*,const int*,const float*,float,float,int,float*,float*);
