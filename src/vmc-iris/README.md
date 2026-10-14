VMC retains 57344 samples, 100000 equilibration generations, 5000 generations
per sample block, seed5551212, and 100 sample blocks by default. Original
propagation and reduction device routines use IRIS dmem tasks. Explicit
reduction offsets preserve the original per-statistic layout. Native HIP
radii, error estimates, and acceptance ratio provide numerical validation.
