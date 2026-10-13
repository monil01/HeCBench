#include <iostream>
#include <cstdlib>
#include "dmem.h"
#include "OptionParser.h"
#include "Utility.h"

using namespace std;

// Forward Declarations
void addBenchmarkSpecOptions(OptionParser &op);
void RunBenchmark(OptionParser &op);

// ****************************************************************************
// Function: main
//
// Purpose:
//   The main function takes care of initialization (device and MPI),  then
//   performs the benchmark and prints results.
//
// Arguments:
//
//
// Programmer: Jeremy Meredith
// Creation:
//
// Modifications:
//   Jeremy Meredith, Wed Nov 10 14:20:47 EST 2010
//   Split timing reports into detailed and summary.  For serial code, we
//   report all trial values, and for parallel, skip the per-process vals.
//   Also detect and print outliers from parallel runs.
//
// ****************************************************************************
int main(int argc, char *argv[])
{
    int ret = 0;
    iris::Platform platform;
    if (platform.init(&argc, &argv, true) != IRIS_SUCCESS) return 1;
    int selected = require_amd_hip();
    if (selected < 0) return 1;
    iris_device_set_default(selected);
    // IRIS initialization reseeds libc RNG; source starts with implicit seed 1.
    srandom(1);

    try
    {

        // Get args
        OptionParser op;

        //Add shared options to the parser
        op.addOption("device", OPT_VECINT, "0",
                "specify device(s) to run on", 'd');
        op.addOption("verbose", OPT_BOOL, "", "enable verbose output", 'v');
        op.addOption("passes", OPT_INT, "10", "specify number of passes", 'n');
        op.addOption("size", OPT_INT, "1", "specify problem size", 's');
        op.addOption("quiet", OPT_BOOL, "", "write minimum necessary to standard output", 'q');

        addBenchmarkSpecOptions(op);

        if (!op.parse(argc, argv))
        {
            op.usage();
            return (op.HelpRequested() ? 0 : 1);
        }

        // Run the benchmark
        RunBenchmark(op);
    }
    catch( std::exception& e )
    {
        std::cerr << e.what() << std::endl;
        ret = 1;
    }
    catch( ... )
    {
        ret = 1;
    }

    if (platform.error_count()) ret = 1;
    platform.finalize();
    return ret;
}
