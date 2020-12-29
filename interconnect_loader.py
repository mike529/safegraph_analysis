import safegraph_parser
import sys
import pickle
import gzip
import os
import argparse
import multiprocessing.pool
import errno
import data_parser

def FullDir(dir_name):    
	return [os.path.join(dir_name, file_name) for file_name in os.listdir(dir_name)]

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("month", help="Required: The month of data to load.")
	parser.add_argument("--tract_type", help="If specified and equal to census_tract compute at a census_tract level otherwise compute at a county level")
	parser.add_argument("--filter_states", help="If specified only compute for these states, useful for faster script or limited memory/storage")
	args = parser.parse_args()

	month = args.month
	if args.tract_type == "census_tract":
		row_key_extractor = safegraph_parser.ConvertRowKeyForTract
	else:
		row_key_extractor = safegraph_parser.ConvertRowKey
	if args.filter_states:
		states = set(args.filter_states.split(','))
	else:
		states = None
	print(states)


	month_files = FullDir("raw_data/placegraph_data/{}".format(month))
	poi_cat_files = FullDir("raw_data/placegraph_data/category_map")
	census_block_stats = "raw_data/placegraph_data/home_panel_summary"
	combined_data_by_state = safegraph_parser.SplitIntoSerializedPoiStates(
		month_files, census_block_stats, poi_cat_files,  
		row_key_extractor=row_key_extractor, filter_states=states)
	print("Storing data")

	for state, combined_data_for_state in combined_data_by_state.iteritems():
		print("Storing data for {}".format(state))
		if args.tract_type == "census_tract":
			data_parser.StoreMarshalInterconnect(combined_data_for_state,  "data/computed_interconnect/{}/{}.marshal".format(month, state))
		else:
			data_parser.StoreMarshalInterconnect(combined_data_for_state, "data/computed_county_interconnect/{}/{}.marshal".format(month, state))


