# Binomial predeclared study

{
  "sizes": [
    16,
    129
  ],
  "num_steps": 2048,
  "block_threads": 128,
  "elements_per_thread": 16,
  "precision": "Float32input/preprocessedcoefficients/storage/compute; independentFloat64terminalexpectation",
  "fixture": "145variedcalloptions spot/strike/maturity/volatility; exactserializedrawparametersandpreprocessedcoefficients",
  "timing": "Sum2 preallocatedkernel1000launchmeans withcontextcompletion; preprocessing/transfersoutside",
  "outputs": "All145prices+4exactoutputguards; inputguardscheckedexactindependently",
  "scope": "Sourcekernelunchanged, original1024option/offloadinclusiveCLI notmeasured; original16localvalues+exchange129 retained"
}

All145pricesfiniteandper-element2e-4+5e-4rel againstindependentFloat64terminalbinomialexpectation. Source2048stepsFP32initialization(__expf nativeversusJuliaexp)andbackwardupdatesaccumulateFP32roundoff; original aggregate5e-4relativecriterion motivates declaredper-price5e-4+smallabsoluteenvelope. Dualreferencesretainidenticalpredeclaredtolerance. Exactinputprefix/guardbytechecksseparate. No continuousBSaccuracytoleranceasserted, discrete-to-continuouserrorreportedseparately.
