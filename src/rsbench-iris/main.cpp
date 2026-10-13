#include "rsbench.h"
#include <iris/iris.hpp>
#include "backend.h"

int main(int argc, char * argv[])
{
	// =====================================================================
	// Initialization & Command Line Read-In
	// =====================================================================

	iris::Platform platform; platform.init(&argc,&argv,1); prove_backend();
	int version = 12;
	
	// Process CLI Fields
	Input input = read_CLI( argc, argv );
	
	// =====================================================================
	// Print-out of Input Summary
	// =====================================================================
	logo(version);
	center_print("INPUT SUMMARY", 79);
	border_print();
	print_input_summary(input);

	// =====================================================================
	// Intialize Simulation Data Structures
	// =====================================================================
	border_print();
	center_print("INITIALIZATION", 79);
	border_print();
	
	auto start = std::chrono::steady_clock::now();
	
	SimulationData SD = initialize_simulation( input );

	auto stop = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

	printf("Initialization Complete. (%.2lf seconds)\n", time * 1e-9);
	
	// =====================================================================
	// Cross Section (XS) Parallel Lookup Simulation Begins
	// =====================================================================
	border_print();
	center_print("SIMULATION", 79);
	border_print();

	unsigned long vhash = 0;

	double kernel_time;

	// Run Simulation
	start = std::chrono::steady_clock::now();

	// Run simulation
	if( input.simulation_method == EVENT_BASED )
	{
		if( input.kernel_id == 0 )
			run_event_based_simulation(input, SD, &vhash, &kernel_time );
		else
		{
			printf("Error: No kernel ID %d found!\n", input.kernel_id);
			exit(1);
		}
	}
	else if( input.simulation_method == HISTORY_BASED )
	{
		printf("History-based simulation not implemented in the code. Instead,\nuse the event-based method with \"-m event\" argument.\n");
		exit(1);
	}

	stop = std::chrono::steady_clock::now();
        time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        double run_time = time * 1e-9;

	// Final hash step
	vhash = vhash % 999983;

	printf("Simulation Complete.\n");

	// =====================================================================
	// Print / Save Results and Exit
	// =====================================================================
	border_print();
	center_print("RESULTS", 79);
	border_print();

	int is_invalid = validate_and_print_results(input, run_time, vhash, kernel_time);

	border_print();

	platform.finalize();
	return is_invalid;
}
