IRIS dmem port retaining original forward-energy kernels, three CLI modes,
small/large width paths, seam selection/removal and resized.bmp output.
All partial buffer writes use rw. Swaps exchange pointers to owned dmem objects.
The native reduction-row pointer offset becomes an explicit int offset with
identical indexing; source device-to-device index reset becomes a bounds-checked
copy kernel. Default small-width dynamic cache becomes equivalent512-int static
storage for supported<=256 width. Experimental COMPUTE_M_SINGLE build mode is
not supported beyond256; it is not the source default.

Local source image helpers are preserved: this checkout's stb_image.h supports
24-bit BMP; its Broadway_tower.jpg fixture is actually a128x128 BMP. Default
filename and100-seam update run arguments are unchanged; inputs must be supplied
in the run directory or through the original filename argument. Native siblings
remain untouched. Validation compares complete resized.bmp artifacts in standard,
update, approximate modes and a320-pixel-wide standard case. CUDA unvalidated.

