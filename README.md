# safegraph_analysis

This module is used to train and simulate the evolution of COVID data for individual states.

In order to build and train models you will need access to the safegraph data.

The safegraph data itself is intentionally not included in the repository. Without the data you can only run model training on the stored precomputed extracts.

# Computing Safegraph Extracts

In order to compute the safegraph data you will need the following.

   * [Monthly Place Pattern](https://docs.safegraph.com/docs/places-schema#section-patterns) data for the month you want to generate. This data should have the .csv.gz files stored in raw_data/placegraph_data/${MONTH}/
   * [Category](https://docs.safegraph.com/docs#section-core-places) data for all the US POIs. This data should have the .csv.gz files stored in raw_data/placegraph_data/category_map
   * [Home Panel Summary](https://www.safegraph.com/neighborhood-patterns) the home_panel_summary file should be stored in raw_data/placegraph_data/home_panel_summary.

 After all the files are stored in those locations you can use the interconnect_loader script to compute the data.

 Run this script from the root directory of the repository

 ```bash
	python interconnect_loader.py july
 ```

 This will load all the county level data for the month of July and store the output files in the desired locations.

There are two optional arguments which can be specified in addition.
	* tract_type: Specify --tract_type=census_tract to compute data at the census tract level instead of the county level.
	* filter_states: Specify a comma separated list of state abbreviations to 
This script will take a long time and use a lot of memory.

If you want to load data for a single state instead

```bash
	./load_state_data.sh sd
```

# Training parameters:

In order to determine the best transmission parameters to fit the course of an outbreak you can run 

```bash
python model_trainer.py may june sd
```

This will load the computed monthly data for the specified state over the range of months specified. 
It will then attempt to fit the county level total and current infection numbers on a monthly basis.

This script will output the top 10 options along with the error.
It will also output a formatted string form of the paramaters which can be used with the next script.


# Running a scenario.

To compute the actual results we can take the parameters computed in the earlier script (or other parameters which we want to measure)
and use the script

```bash
python model_computer.py may july sd "0.3808,0.1599,0.0037,0.0019,0.0003,0.0024,0.0017"
```

Right now this runs the simulation and prints out some basic summary statistics.
This will be adapted to add output to a csv.

