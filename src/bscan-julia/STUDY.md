# bscan verified configuration

bscan_independent_oneblock_allsixN_32to1024_repeat10_v1

Originalnative redundantgrid globaloutputrace notadmitted; onlysingleblockpercase verified/timed.

Native warpballot/volatilewarp scan assumeswarpordering; V100numericalverification doesnotprove crossarchitecturememorymodelcorrectness.

HistoricalJulia perthread quadraticprefix vsnewKA/JACC HillisSteele workgroup scan, nativewarpprefix; algorithmadaptationexplicit.

KA allocatesstatic1024Int32elements allcases; JACC dynamic4*Nbytes+templatecopy; not resourceidentical.

All6 blocksize cases, ONEblock/case forALLmodels. Original30240blocks concurrentlywrite sameoutputaddresses (globalwrite race); named independentblockprimitive study avoids that race; not originalredundantgrid performance.

Five in-process trials, all outputs checked against both references and the configured CPU oracle. First trial time excluded; arithmetic mean of last four. Raw samples/configuration/provenance remain linked in per-variant verification.json. Partial token estimates are not total usage.
