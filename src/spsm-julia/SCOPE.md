# Sparse lower-triangular multiple-RHS solve

HistoricalJuliaunrelatedMCPRincludeS1. RestoredoriginalvendorSpSMAPI/analysis/LOWER/NONUNIT/ROW/NONTRANSPOSE/alpha1/F32/Int32. SourcepreloadsGPUoutputwithtruthsolution; namedverificationpoisonsCafterwarm/analysis/everytrialoutsideROI, strongeroutputinstrumentationnotmathematicalnativecorrection. CPUno-opcounterfactualnotobservedsourceGPUfailure.

Diag2or4/offdiag±1/64, truthXinteger/8 andB integer/512. Atmost7offentriesperrow, strictdominancebound7/64<mindiag2. Boundmaxabsproductsumunits1/512<2^23; exactpower2division/partialsumstratum. CompactFractionforwardoracle/fullknownsolution/fullCSRresidualforallactual20files. Arbitraryfloat/originalRNGconditioning/solverconvergence unmeasured.

Source20082rows/128RHS/150616nnzGPUfootprint retainednamedcontrolledCSR, plus1/17/33/257/oddRHS+zeroRHScontrol. Directsparsegenerationavoidsdensehostinitializer outsideROI; originalRNG/conditioning/repeat100unmeasured, namedR10. Originalnnz<rowsCLIclampmayoutgrowhostpreallocation staticriskoutofscope; allnamednnz>=rows.

SameactualcuSPARSE12001binary/solvefunction andLOWER/NONUNIT/ROW/NONTRANSPOSE/alpha1/DEFAULT/HOSTmode/legacy0. Analysis/descriptors/workspaceoutsideROI; GPUoutputpoisonreset/copyprefix/capture/checkoutsideROI; percase10solve+completion. All3hostvendorinterop/0authoredkernels, noKA/JACCGPUlowering/no causalabstractionranking. Vendorworkspacebytesmatchedall4outsideROI; unprofilednotDRAMtraffic/FLOProofline.

Prospectiveinclusiveworkflow/referencewindows includecontext/cache/paper/audits/coordination. KA/JACCwindowsbeginBEFOREeachownhostwrapperdraft/evaluation, sharedvendorhelper alreadyprepared/referenceverified. Thesecompletewrapperactivitywindows arenotexclusive/billedtranslationcost. Sharedworkflowoverlapsphasesdonotadd; sourcechars/4partialsizeproxyseparate/historicalunknown.

## Original notice

/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
