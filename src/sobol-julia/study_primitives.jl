# /*
# * Portions Copyright (c) 1993-2015 NVIDIA Corporation.  All rights reserved.
# * Please refer to the NVIDIA end user license agreement (EULA) associated
# * with this source code for terms and conditions that govern your use of
# * this software. Any use, reproduction, disclosure, or distribution of
# * this software and related documentation outside the terms of the EULA
# * is strictly prohibited.
# * 
# * Portions Copyright (c) 2009 Mike Giles, Oxford University.  All rights reserved.
# * Portions Copyright (c) 2008 Frances Y. Kuo and Stephen Joe.  All rights reserved.
# *
# * Sobol Quasi-random Number Generator example
# *
# * Based on CUDA code submitted by Mike Giles, Oxford University, United Kingdom
# * http://people.maths.ox.ac.uk/~gilesm/
# *
# * and C code developed by Stephen Joe, University of Waikato, New Zealand
# * and Frances Kuo, University of New South Wales, Australia
# * http://web.maths.unsw.edu.au/~fkuo/sobol/
# *
# * For theoretical background see:
# *
# * P. Bratley and B.L. Fox.
# * Implementing Sobol's quasirandom sequence generator
# * http://portal.acm.org/citation.cfm?id=42288
# * ACM Trans. on Math. Software, 14(1):88-100, 1988
# *
# * S. Joe and F. Kuo.
# * Remark on algorithm 659: implementing Sobol's quasirandom sequence generator.
# * http://portal.acm.org/citation.cfm?id=641879
# * ACM Trans. on Math. Software, 29(1):49-57, 2003
# *
# */
# 
# First100primitive rows extracted verbatim as integers; polynomial recurrence computed on host.
const STUDY_PRIMITIVES=[
(0,UInt32(0),UInt32[]),
(1,UInt32(0),UInt32[1]),
(2,UInt32(1),UInt32[1,3]),
(3,UInt32(1),UInt32[1,3,1]),
(3,UInt32(2),UInt32[1,1,1]),
(4,UInt32(1),UInt32[1,1,3,3]),
(4,UInt32(4),UInt32[1,3,5,13]),
(5,UInt32(2),UInt32[1,1,5,5,17]),
(5,UInt32(4),UInt32[1,1,5,5,5]),
(5,UInt32(7),UInt32[1,1,7,11,19]),
(5,UInt32(11),UInt32[1,1,5,1,1]),
(5,UInt32(13),UInt32[1,1,1,3,11]),
(5,UInt32(14),UInt32[1,3,5,5,31]),
(6,UInt32(1),UInt32[1,3,3,9,7,49]),
(6,UInt32(13),UInt32[1,1,1,15,21,21]),
(6,UInt32(16),UInt32[1,3,1,13,27,49]),
(6,UInt32(19),UInt32[1,1,1,15,7,5]),
(6,UInt32(22),UInt32[1,3,1,15,13,25]),
(6,UInt32(25),UInt32[1,1,5,5,19,61]),
(7,UInt32(1),UInt32[1,3,7,11,23,15,103]),
(7,UInt32(4),UInt32[1,3,7,13,13,15,69]),
(7,UInt32(7),UInt32[1,1,3,13,7,35,63]),
(7,UInt32(8),UInt32[1,3,5,9,1,25,53]),
(7,UInt32(14),UInt32[1,3,1,13,9,35,107]),
(7,UInt32(19),UInt32[1,3,1,5,27,61,31]),
(7,UInt32(21),UInt32[1,1,5,11,19,41,61]),
(7,UInt32(28),UInt32[1,3,5,3,3,13,69]),
(7,UInt32(31),UInt32[1,1,7,13,1,19,1]),
(7,UInt32(32),UInt32[1,3,7,5,13,19,59]),
(7,UInt32(37),UInt32[1,1,3,9,25,29,41]),
(7,UInt32(41),UInt32[1,3,5,13,23,1,55]),
(7,UInt32(42),UInt32[1,3,7,3,13,59,17]),
(7,UInt32(50),UInt32[1,3,1,3,5,53,69]),
(7,UInt32(55),UInt32[1,1,5,5,23,33,13]),
(7,UInt32(56),UInt32[1,1,7,7,1,61,123]),
(7,UInt32(59),UInt32[1,1,7,9,13,61,49]),
(7,UInt32(62),UInt32[1,3,3,5,3,55,33]),
(8,UInt32(14),UInt32[1,3,1,15,31,13,49,245]),
(8,UInt32(21),UInt32[1,3,5,15,31,59,63,97]),
(8,UInt32(22),UInt32[1,3,1,11,11,11,77,249]),
(8,UInt32(38),UInt32[1,3,1,11,27,43,71,9]),
(8,UInt32(47),UInt32[1,1,7,15,21,11,81,45]),
(8,UInt32(49),UInt32[1,3,7,3,25,31,65,79]),
(8,UInt32(50),UInt32[1,3,1,1,19,11,3,205]),
(8,UInt32(52),UInt32[1,1,5,9,19,21,29,157]),
(8,UInt32(56),UInt32[1,3,7,11,1,33,89,185]),
(8,UInt32(67),UInt32[1,3,3,3,15,9,79,71]),
(8,UInt32(70),UInt32[1,3,7,11,15,39,119,27]),
(8,UInt32(84),UInt32[1,1,3,1,11,31,97,225]),
(8,UInt32(97),UInt32[1,1,1,3,23,43,57,177]),
(8,UInt32(103),UInt32[1,3,7,7,17,17,37,71]),
(8,UInt32(115),UInt32[1,3,1,5,27,63,123,213]),
(8,UInt32(122),UInt32[1,1,3,5,11,43,53,133]),
(9,UInt32(8),UInt32[1,3,5,5,29,17,47,173,479]),
(9,UInt32(13),UInt32[1,3,3,11,3,1,109,9,69]),
(9,UInt32(16),UInt32[1,1,1,5,17,39,23,5,343]),
(9,UInt32(22),UInt32[1,3,1,5,25,15,31,103,499]),
(9,UInt32(25),UInt32[1,1,1,11,11,17,63,105,183]),
(9,UInt32(44),UInt32[1,1,5,11,9,29,97,231,363]),
(9,UInt32(47),UInt32[1,1,5,15,19,45,41,7,383]),
(9,UInt32(52),UInt32[1,3,7,7,31,19,83,137,221]),
(9,UInt32(55),UInt32[1,1,1,3,23,15,111,223,83]),
(9,UInt32(59),UInt32[1,1,5,13,31,15,55,25,161]),
(9,UInt32(62),UInt32[1,1,3,13,25,47,39,87,257]),
(9,UInt32(67),UInt32[1,1,1,11,21,53,125,249,293]),
(9,UInt32(74),UInt32[1,1,7,11,11,7,57,79,323]),
(9,UInt32(81),UInt32[1,1,5,5,17,13,81,3,131]),
(9,UInt32(82),UInt32[1,1,7,13,23,7,65,251,475]),
(9,UInt32(87),UInt32[1,3,5,1,9,43,3,149,11]),
(9,UInt32(91),UInt32[1,1,3,13,31,13,13,255,487]),
(9,UInt32(94),UInt32[1,3,3,1,5,63,89,91,127]),
(9,UInt32(103),UInt32[1,1,3,3,1,19,123,127,237]),
(9,UInt32(104),UInt32[1,1,5,7,23,31,37,243,289]),
(9,UInt32(109),UInt32[1,1,5,11,17,53,117,183,491]),
(9,UInt32(122),UInt32[1,1,1,5,1,13,13,209,345]),
(9,UInt32(124),UInt32[1,1,3,15,1,57,115,7,33]),
(9,UInt32(137),UInt32[1,3,1,11,7,43,81,207,175]),
(9,UInt32(138),UInt32[1,3,1,1,15,27,63,255,49]),
(9,UInt32(143),UInt32[1,3,5,3,27,61,105,171,305]),
(9,UInt32(145),UInt32[1,1,5,3,1,3,57,249,149]),
(9,UInt32(152),UInt32[1,1,3,5,5,57,15,13,159]),
(9,UInt32(157),UInt32[1,1,1,11,7,11,105,141,225]),
(9,UInt32(167),UInt32[1,3,3,5,27,59,121,101,271]),
(9,UInt32(173),UInt32[1,3,5,9,11,49,51,59,115]),
(9,UInt32(176),UInt32[1,1,7,1,23,45,125,71,419]),
(9,UInt32(181),UInt32[1,1,3,5,23,5,105,109,75]),
(9,UInt32(182),UInt32[1,1,7,15,7,11,67,121,453]),
(9,UInt32(185),UInt32[1,3,7,3,9,13,31,27,449]),
(9,UInt32(191),UInt32[1,3,1,15,19,39,39,89,15]),
(9,UInt32(194),UInt32[1,1,1,1,1,33,73,145,379]),
(9,UInt32(199),UInt32[1,3,1,15,15,43,29,13,483]),
(9,UInt32(218),UInt32[1,1,7,3,19,27,85,131,431]),
(9,UInt32(220),UInt32[1,3,3,3,5,35,23,195,349]),
(9,UInt32(227),UInt32[1,3,3,7,9,27,39,59,297]),
(9,UInt32(229),UInt32[1,1,3,9,11,17,13,241,157]),
(9,UInt32(230),UInt32[1,3,7,15,25,57,33,189,213]),
(9,UInt32(234),UInt32[1,1,7,1,9,55,73,83,217]),
(9,UInt32(236),UInt32[1,3,3,13,19,27,23,113,249]),
(9,UInt32(241),UInt32[1,3,5,3,23,43,3,253,479]),
(9,UInt32(244),UInt32[1,1,5,5,11,5,45,117,217]),
]
