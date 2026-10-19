// Exact source device functions; see extraction_manifest.json for spans and hashes.
template<typename scalar_t, typename index_t>
                C10_LAUNCH_BOUNDS_1(256) __global__ void _braycurtis_forward_kernel_impl(
                        index_t n_kernels,
                        const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> x1,
                        const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> x2,
                        at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> output) {
                    CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                        index_t j = index % x2.size(1);
                        index_t i = (index / x2.size(1)) % x1.size(1);
                        index_t b = index / (x2.size(1) * x1.size(1));

                        scalar_t diff = 0, sum = 0;
                        for (index_t k = 0; k < x1.size(2); k++) {
                            diff += fabs(x1[b][i][k] - x2[b][j][k]);
                            sum += fabs(x1[b][i][k] + x2[b][j][k]);
                        }
                        output[b][i][j] = diff / sum;
                    }
                }

template<typename scalar_t, typename index_t>
                C10_LAUNCH_BOUNDS_1(256) __global__ void _braycurtis_forward_components_kernel_impl(
                        index_t n_kernels,
                        const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> x1,
                        const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> x2,
                        at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> sd,
                        at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> ss) {
                    CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                        index_t j = index % x2.size(1);
                        index_t i = (index / x2.size(1)) % x1.size(1);
                        index_t b = index / (x2.size(1) * x1.size(1));

                        scalar_t diff = 0, sum = 0;
                        for (index_t k = 0; k < x1.size(2); k++) {
                            diff += fabs(x1[b][i][k] - x2[b][j][k]);
                            sum += fabs(x1[b][i][k] + x2[b][j][k]);
                        }
                        sd[b][i][j] = diff;
                        ss[b][i][j] = sum;
                    }
                }

template<typename scalar_t, typename index_t>
                C10_LAUNCH_BOUNDS_1(256) __global__ void _braycurtis_backward_x1_kernel_impl(
                        index_t n_kernels,
                        const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> grad_output,
                        const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> sd,
                        const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> ss,
                        const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> x1,
                        const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> x2,
                        at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> grad_x1) {
                    CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                        index_t k = index % x1.size(2);
                        index_t i = (index / x1.size(2)) % x1.size(1);
                        index_t b = index / (x1.size(2) * x1.size(1));

                        scalar_t val = 0, diff, sum;
                        for (index_t j = 0; j < x2.size(1); j++) {
                            diff = x1[b][i][k] - x2[b][j][k];
                            sum = x1[b][i][k] + x2[b][j][k];
                            val += grad_output[b][i][j] *
                                    (m_signum(diff) / ss[b][i][j] -
                                    m_signum(sum) * sd[b][i][j] / ss[b][i][j] / ss[b][i][j]);
                        }
                        grad_x1[b][i][k] = val;
                    }
                }

template<typename scalar_t, typename index_t>
                C10_LAUNCH_BOUNDS_1(256) __global__ void _braycurtis_backward_x2_kernel_impl(
                        index_t n_kernels,
                        const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> grad_output,
                        const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> sd,
                        const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> ss,
                        const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> x1,
                        const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> x2,
                        at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> grad_x2) {
                    CUDA_1D_KERNEL_LOOP(index, n_kernels) {
                        index_t k = index % x2.size(2);
                        index_t j = (index / x2.size(2)) % x2.size(1);
                        index_t b = index / (x2.size(2) * x2.size(1));

                        scalar_t val = 0, diff, sum;
                        for (index_t i = 0; i < x1.size(1); i++) {
                            diff = x1[b][i][k] - x2[b][j][k];
                            sum = x1[b][i][k] + x2[b][j][k];
                            val += grad_output[b][i][j] *
                                   (-m_signum(diff) / ss[b][i][j] -
                                    m_signum(sum) * sd[b][i][j] / ss[b][i][j] / ss[b][i][j]);
                        }
                        grad_x2[b][j][k] = val;
                    }
                }
