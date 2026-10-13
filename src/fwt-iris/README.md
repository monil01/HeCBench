# Fast Walsh transform IRIS
Preserves repeat argument/defaultrun100, data2^23/kernel2^7, srand123,
radix4/radix2 butterfly and modulation geometry, CPU dyadic convolution and
L2<1e-6 PASS criterion. DMem in-place transforms use iris_rw on both aliased
arguments; zero padding/data reset each repeat explicitly refreshed with
DMem.update(host). Dynamic shared memory replaced by static2048-float array
(max elementary size,8KiB); arithmetic unchanged, small transforms reserve
more shared memory. FWT_VALIDATION_LOG2_DATA optionally bounds data power
(kernel power remains7). FWT_VALIDATION_DUMP optionally saves every output.
