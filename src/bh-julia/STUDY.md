# Barnes-Hut reference semantic blocker

Native ECL-BHv4.5 (sourceheader: starclustergravitational simulation) builds/summarizes/sorts an octree, applies a warpcollective cellopeningcriterion (`itolsq=4`,theta0.5), calculates softenedgravity, then separately integrates. Preserved Julia `direct_step_kernel!` instead scans allbodies and updatespositions inside that samekernel. It has no tree oropeningcriterion. Otherthreads may readpositionswhile they are beingwritten. This is source-established S1 algorithmreplacement and read/writehazard, not a measured frameworkcompile/runtime failure.

HistoricalJulia prints runtime only, with no numericalverifier. No existingPASS establishes a faithful Barnes-Hut reference. KA/JACC targettranslation remains unattempted/referenceblocked (zero targetcandidate/checks); no frameworkfeatureunsupported conclusion or inventedtimings.

Resume by restoring realtreepipeline inCUDA.jl with oldsourcepreserved and separatelycosted, fixedinputs and fullstate/CPUphysics numericalcontract, then KA followed byJACC. Deviceatommemoryordering/warpcollectives mustbe audited against native; no denseforce substitute under theBarnes-Hut label. Exacthashedsourceevidence and phaseevents are in ontology studyinputs/state.
