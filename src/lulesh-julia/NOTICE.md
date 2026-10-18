# LULESH provenance and license

Original LULESH2.0 numerical/initialization sources retain the following LLNL BSD and DOE notice in reference_original. Derived source_cpu_math.h/source_volume.cpp/study_device.jl preserve numerical bodies; the owned adapter changes no source device arithmetic.

```text

   Copyright (c) 2010-2013.
   Lawrence Livermore National Security, LLC.
   Produced at the Lawrence Livermore National Laboratory.
   LLNL-CODE-461231
   All rights reserved.

   This file is part of LULESH, Version 2.0.
   Please also read this link -- http://www.opensource.org/licenses/index.php

//////////////
DIFFERENCES BETWEEN THIS VERSION (2.x) AND EARLIER VERSIONS:
 * Addition of regions to make work more representative of multi-material codes
 * Default size of each domain is 30^3 (27000 elem) instead of 45^3. This is
 more representative of our actual working set sizes
 * Single source distribution supports pure serial, pure OpenMP, MPI-only,
 and MPI+OpenMP
 * Addition of ability to visualize the mesh using VisIt
https://wci.llnl.gov/codes/visit/download.html
 * Various command line options (see ./lulesh2.0 -h)
 -q              : quiet mode - suppress stdout
 -i <iterations> : number of cycles to run
 -s <size>       : length of cube mesh along side
 -r <numregions> : Number of distinct regions (def: 11)
 -b <balance>    : Load balance between regions of a domain (def: 1)
 -c <cost>       : Extra cost of more expensive regions (def: 1)
 -f <filepieces> : Number of file parts for viz output (def: np/9)
 -p              : Print out progress
 -v              : Output viz file (requires compiling with -DVIZ_MESH
 -h              : This message

 printf("Usage: %s [opts]\n", execname);
 printf(" where [opts] is one or more of:\n");
 printf(" -q              : quiet mode - suppress all stdout\n");
 printf(" -i <iterations> : number of cycles to run\n");
 printf(" -s <size>       : length of cube mesh along side\n");
 printf(" -r <numregions> : Number of distinct regions (def: 11)\n");
 printf(" -b <balance>    : Load balance between regions of a domain (def: 1)\n");
 printf(" -c <cost>       : Extra cost of more expensive regions (def: 1)\n");
 printf(" -f <numfiles>   : Number of files to split viz dump into (def: (np+10)/9)\n");
 printf(" -p              : Print out progress\n");
 printf(" -v              : Output viz file (requires compiling with -DVIZ_MESH\n");
 printf(" -h              : This message\n");
 printf("\n\n");

 *Notable changes in LULESH 2.0

 * Split functionality into different files
 lulesh.cc - where most (all?) of the timed functionality lies
 lulesh-comm.cc - MPI functionality
 lulesh-init.cc - Setup code
 lulesh-viz.cc  - Support for visualization option
 lulesh-util.cc - Non-timed functions
 *
 * The concept of "regions" was added, although every region is the same ideal
 *    gas material, and the same sedov blast wave problem is still the only
 *    problem its hardcoded to solve.
 * Regions allow two things important to making this proxy app more representative:
 *   Four of the LULESH routines are now performed on a region-by-region basis,
 *     making the memory access patterns non-unit stride
 *   Artificial load imbalances can be easily introduced that could impact
 *     parallelization strategies.
 * The load balance flag changes region assignment.  Region number is raised to
 *   the power entered for assignment probability.  Most likely regions changes
 *   with MPI process id.
 * The cost flag raises the cost of ~45% of the regions to evaluate EOS by the
 *   entered multiple. The cost of 5% is 10x the entered multiple.
 * MPI and OpenMP were added, and coalesced into a single version of the source
 *   that can support serial builds, MPI-only, OpenMP-only, and MPI+OpenMP
* Added support to write plot files using "poor mans parallel I/O" when linked
*   with the silo library, which in turn can be read by VisIt.
* Enabled variable timestep calculation by default (courant condition), which
*   results in an additional reduction.
* Default domain (mesh) size reduced from 45^3 to 30^3
* Command line options to allow numerous test cases without needing to recompile
* Performance optimizations and code cleanup beyond LULESH 1.0
* Added a "Figure of Merit" calculation (elements solved per microsecond) and
*   output in support of using LULESH 2.0 for the 2017 CORAL procurement
  *
* Possible Differences in Final Release (other changes possible)
  *
  * High Level mesh structure to allow data structure transformations
  * Different default parameters
  * Minor code performance changes and cleanup

  TODO in future versions
  * Add reader for (truly) unstructured meshes, probably serial only
  * CMake based build system

  //////////////

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

  * Redistributions of source code must retain the above copyright
  notice, this list of conditions and the disclaimer below.

  * Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the disclaimer (as noted below)
  in the documentation and/or other materials provided with the
  distribution.

  * Neither the name of the LLNS/LLNL nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
  THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
      BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
      DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
      NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


  Additional BSD Notice

  1. This notice is required to be provided under our contract with the U.S.
  Department of Energy (DOE). This work was produced at Lawrence Livermore
  National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.

  2. Neither the United States Government nor Lawrence Livermore National
  Security, LLC nor any of their employees, makes any warranty, express
  or implied, or assumes any liability or responsibility for the accuracy,
  completeness, or usefulness of any information, apparatus, product, or
  process disclosed, or represents that its use would not infringe
  privately-owned rights.

  3. Also, reference herein to any specific commercial products, process, or
  services by trade name, trademark, manufacturer or otherwise does not
  necessarily constitute or imply its endorsement, recommendation, or
  favoring by the United States Government or Lawrence Livermore National
  Security, LLC. The views and opinions of authors expressed herein do not
  necessarily state or reflect those of the United States Government or
  Lawrence Livermore National Security, LLC, and shall not be used for
  advertising or product endorsement purposes.

  
```
