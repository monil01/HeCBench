//=====================================================================
//  MAIN FUNCTION
//=====================================================================

double master(fp timeinst,
    fp* initvalu,
    fp* parameter,
    fp* finavalu,
    fp* com,

    iris::DMem & d_initvalu,
    iris::DMem & d_finavalu,
    iris::DMem & d_params,
    iris::DMem & d_com)
{

  //=====================================================================
  //  VARIABLES
  //=====================================================================

  // counters
  int i;

  // offset pointers
  int initvalu_offset_ecc;
  int initvalu_offset_Dyad;
  int initvalu_offset_SL;
  int initvalu_offset_Cyt;


  //=====================================================================
  //  execute ECC&CAM kernel - it runs ECC and CAMs in parallel
  //=====================================================================

  int d_initvalu_mem;
  d_initvalu_mem = EQUATIONS * sizeof(fp);
  int d_finavalu_mem;
  d_finavalu_mem = EQUATIONS * sizeof(fp);
  int d_params_mem;
  d_params_mem = PARAMETERS * sizeof(fp);
  int d_com_mem;
  d_com_mem = 3 * sizeof(fp);

#ifdef DEBUG
  for (int i = 0; i < EQUATIONS; i++)
    printf("initvalu %d %f\n", i, initvalu[i]);
  for (int i = 0; i < PARAMETERS; i++)
    printf("params %d %f\n", i, parameter[i]);
  printf("\n");
#endif

  iris_check(d_initvalu.update(initvalu));
  iris_check(d_params.update(parameter));
  iris_check(d_finavalu.update(finavalu));
  iris_check(d_com.update(com));
  // CUDA master accepts float time but the device kernel takes int.
  int kernel_timeinst = static_cast<int>(timeinst);
  auto start = std::chrono::steady_clock::now();
  iris::Task task;
  void *args[] = { &kernel_timeinst, &d_initvalu, &d_finavalu, &d_params, &d_com };
  int info[] = { sizeof(kernel_timeinst), iris_r, iris_w, iris_r, iris_w };
  size_t gws[] = { 2 * NUMBER_THREADS }, lws[] = { NUMBER_THREADS };
  iris_check(task.kernel("kernel", 1, nullptr, gws, lws, 5, args, info));
  iris_check(task.flush_out(d_finavalu));
  iris_check(task.flush_out(d_com));
  iris_check(task.submit(iris_gpu, nullptr, true));
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  // Optional first RHS capture for independent equation validation.
  static bool captured = false;
  if (!captured) if (const char* path = getenv("MYOCYTE_RHS_DUMP")) {
    FILE* f = fopen(path, "wb");
    if (!f) exit(1);
    fwrite(&kernel_timeinst, sizeof(int), 1, f);
    fwrite(initvalu, sizeof(fp), EQUATIONS, f);
    fwrite(parameter, sizeof(fp), PARAMETERS, f);
    fwrite(finavalu, sizeof(fp), EQUATIONS, f);
    fwrite(com, sizeof(fp), 3, f);
    fclose(f);
    captured = true;
  }

#ifdef DEBUG
  for (int i = 0; i < EQUATIONS; i++)
    printf("finavalu %d %f\n", i, finavalu[i]);
  for (int i = 0; i < 3; i++)
    printf("%f ", com[i]);
  printf("\n");

#endif

  //=====================================================================
  //  FINAL KERNEL
  //=====================================================================

  initvalu_offset_ecc = 0;
  initvalu_offset_Dyad = 46;
  initvalu_offset_SL = 61;
  initvalu_offset_Cyt = 76;

  kernel_fin(
      initvalu,
      initvalu_offset_ecc,
      initvalu_offset_Dyad,
      initvalu_offset_SL,
      initvalu_offset_Cyt,
      parameter,
      finavalu,
      com[0],
      com[1],
      com[2]);

  //=====================================================================
  //  COMPENSATION FOR NANs and INFs
  //=====================================================================

  for(i=0; i<EQUATIONS; i++){
    if (isnan(finavalu[i])){
      finavalu[i] = 0.0001;                        // for NAN set rate of change to 0.0001
    }
    else if (isinf(finavalu[i])){
      finavalu[i] = 0.0001;                        // for INF set rate of change to 0.0001
    }
  }

  return time;
}
