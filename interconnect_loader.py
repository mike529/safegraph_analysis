import safegraph_parser
import sys
import pickle
import gzip
import os
import argparse
import multiprocessing.pool
import errno

def FullDir(dir_name):    
	return [os.path.join(dir_name, file_name) for file_name in os.listdir(dir_name)]




def StorePickleFile(pickle_data, file_name):
	if not os.path.exists(os.path.dirname(file_name)):
	    try:
	        os.makedirs(os.path.dirname(file_name))
	    except OSError as exc: # Guard against race condition
	        if exc.errno != errno.EEXIST:
	            raise

	print("Writing to {}".format(file_name))

	if file_name.endswith('gz'):
		with gzip.open(file_name, 'w') as g:
			pickle.dump(pickle_data, g, protocol=-1)
	else:
		with open(file_name, 'w') as g:
			pickle.dump(pickle_data, g, protocol=-1) 



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
	pool = multiprocessing.pool.ThreadPool(8)

	def StoreState(state_tuple):
		state, combined_data_for_state = state_tuple
		StorePickleFile(combined_data_for_state, "data/computed_interconnect/{}/{}.pickle".format(month, state))

	pool.map(StoreState, combined_data_by_state.iteritems())



