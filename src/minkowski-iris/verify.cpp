#ifdef VERIFY
bool ValueSame(float a, float b) {
  return fabsf(a - b) <= 1e-5f;
}

void VerifyResult(float (*a_host)[N], float (*b_host)[K],
                  float (*c_host)[K], float (*c_back)[K],
                  float p, float one_over_p)
{
  int i, j, k;
  for (i = 0; i < M; i++)
    for (j = 0; j < K; j++) c_host[i][j] = 0.0f;

  for (i = 0; i < M; i++) {
    for (k = 0; k < N; k++) {
      for (j = 0; j < K; j++) {
        c_host[i][j] += powf(fabsf(a_host[i][k] - b_host[k][j]), p);
      }
    }
  }
  for (i = 0; i < M; i++)
    for (j = 0; j < K; j++) c_host[i][j] = powf(c_host[i][j], one_over_p);

  bool mismatch_found = false;
  int print_count = 0;
  for (i = 0; i < M; i++) {
    for (j = 0; j < K; j++) {
      if (!ValueSame(c_back[i][j], c_host[i][j])) {
        cout << "Fail - The result is incorrect for element: [" << i << ", "
             << j << "], expected: " << c_host[i][j]
             << ", but found: " << c_back[i][j] << "\n";
        mismatch_found = true;
        print_count++;
        if (print_count == 5) break;
      }
    }
    if (print_count == 5) break;
  }
  cout << (mismatch_found ? "FAIL\n" : "PASS\n");
}
#endif
