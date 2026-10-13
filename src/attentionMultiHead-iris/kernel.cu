extern "C" __global__ void mha(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, int beam_size, int n_steps, int qk_col,
    int v_col, int nhead, float scale, int threshold,
    float* __restrict__ dst) {
  const size_t output_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t output_size = static_cast<size_t>(beam_size) * qk_col;
  if (output_index >= output_size) return;
  const int dim_per_head = qk_col / nhead;
  const int candidate_id = static_cast<int>(output_index / qk_col);
  const int feature = static_cast<int>(output_index % qk_col);
  const int head_id = feature / dim_per_head;
  const int head_feature = feature % dim_per_head;
  float logits[9];
  float max_logit = -1e-20f;
  for (int step = 0; step < n_steps; ++step) {
    float dot = 0.f;
    const int q_offset = candidate_id * qk_col + head_id * dim_per_head;
    const int k_offset = candidate_id * qk_col * n_steps + head_id * dim_per_head + step * qk_col;
    for (int d = 0; d < dim_per_head; ++d) dot += q[q_offset + d] * k[k_offset + d];
    logits[step] = dot * scale;
    max_logit = fmaxf(max_logit, logits[step]);
  }
  float denominator = 0.f;
  for (int step = 0; step < n_steps; ++step) {
    logits[step] = expf(fmaxf(logits[step] - max_logit, -static_cast<float>(threshold)));
    denominator += logits[step];
  }
  float result = 0.f;
  const int v_offset = candidate_id * v_col * n_steps + head_id * dim_per_head + head_feature;
  for (int step = 0; step < n_steps; ++step)
    result += (logits[step] / denominator) * v[v_offset + step * v_col];
  dst[output_index] = result;
}
