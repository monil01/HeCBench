# Hotspot3D IRIS
Preserves six positional arguments, 64x4 blocks, thermal constants, helper input/
output layout, CPU RMS check, ping-pong selection including original even-iteration
output selection. Run defaults512 rows/cols,8 layers,5000 iterations and original
input/output filenames retained. DMem owns three arrays, stencil read/write modes
preserve device intermediates. Validation compares complete output artifacts and
RMS against HIP on identical deterministic bounded fixtures for odd/even iterations.
