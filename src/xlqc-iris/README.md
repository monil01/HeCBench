XLQC retains the original GSL host SCF and the original HIP Rys integral
routines for sp/dp. IRIS dmem manages all device buffers and both J/K tasks.
Set GSL_INC and GSL_LIB for a local GSL install; DATA_DIR supplies the original
geometry and basis files. GPU integral computation has no CPU fallback.
