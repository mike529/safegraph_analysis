import csv
import json
import collections
import sys
import os
import shapefile
import scipy.sparse
import numpy
import gzip
import model_evolution
import data_parser

csv.field_size_limit(sys.maxsize)

def GetCensusBlockStats(file_name, row_key_extractor):
	census_block_stats = {}
	with open(file_name) as f:
		csv_dict_reader = csv.DictReader(f)
		for i, row in enumerate(csv_dict_reader):
			converted = row_key_extractor(row['census_block_group'])
			existing = census_block_stats.get(converted, {})
			existing_pop = existing.get('population', 0)

			census_block_stats[converted] = {'state': RowState(row), 'population': int(row['number_devices_residing']) + existing_pop}
	return census_block_stats	


class NeighborhoodStats():
 	def __init__(self):
 		self.neighbors = {}
 		self.total_pings = 0
 		self.population = None
 		self.total_neighbors = 0.0

	def AddInteractions(self, neighbor, square_pings):
		self.neighbors[neighbor] = square_pings
		self.total_neighbors += square_pings

	def ReduceNeighbors(self, min_fraction):
		min_required = self.total_neighbors * min_fraction
		new_neighbors = {}
		new_total = 0
		for neighbor, total in self.neighbors.iteritems():
			if total > min_required:
				new_neighbors[neighbor] = total
				new_total += total
		self.neighbors = new_neighbors
		self.total_neighbors = new_total


			



def GetCategory(naics_code):
	
	top_level_code = naics_code / 10000
	if top_level_code == 61:
		return "EDUCATION"
	elif top_level_code == 72:
		return "RESTAURANT"
	elif top_level_code == 62:
		return "HEALTH_CARE"
	elif naics_code / 1000 == 813:
		return "RELIGION"
	else:
		return "OTHER"



def ConvertRowKey(row_key):
	# Uses the county as the row key.
	# Modify to use more of the key, for more granular data.
	return row_key[:5]

def ConvertRowKeyForTract(row_key):
	# Uses the census tract as the row_key
	return row_key[:-1]	

def RowState(row):
	# Uses the state modify to split less granularly
	return row['state']


def MarkKey(state, visit_category, final_key, value, dictionary):
	current = dictionary
	if state not in current:
		current[state] = {}
	current = current[state]
	if visit_category not in current:
		current[visit_category] = {}
	current = current[visit_category]
	if final_key not in current:
		current[final_key] = 0
	current[final_key] += value

def MapPoiToCategory(file_names, category_extractor=GetCategory):
	poi_to_category = {}
	for file_name in file_names:
		print("Parsing file: {}".format(file_name))
		with gzip.open(file_name) as f:
			csv_dict_reader = csv.DictReader(f)
			for i, row in enumerate(csv_dict_reader):
				poi_to_category[row['safegraph_place_id']] = category_extractor(int(row['naics_code'] or 0))
	return poi_to_category

def LoadPoiPatterns(file_names, census_block_stats, poi_to_category, row_key_extractor, filter_states):
	pings_by_neighborhood = {}
	visits = {}
	def GetId(area):
		converted = row_key_extractor(area)
		return converted

	def GetState(area):
		census_block_stat = census_block_stats.get(row_key_extractor(area))
		if not census_block_stat:
			return None
		return census_block_stat['state']


	total_count = 0
	for file_name in file_names:
		print("Parsing file: {}".format(file_name))
		with gzip.open(file_name) as f:
			csv_dict_reader = csv.DictReader(f)
			for i, row in enumerate(csv_dict_reader):
				total_count += 1
				if i % 100000 == 0:
					print("Loading row #{}: POI:{}".format(i, row['location_name']))
				poi_state = GetState(row['poi_cbg'])
				poi_id = total_count
				poi_type = poi_to_category.get(row['safegraph_place_id'], 'OTHER')
				median_dwell = float(row['median_dwell'])
				for source_location, source_ping_count in json.loads(row['visitor_home_cbgs']).iteritems():
					source_state = GetState(source_location)
					if filter_states and source_state not in filter_states:
						continue
					source_id = GetId(source_location)
					MarkKey(source_state, poi_type, source_id, (source_ping_count * median_dwell), pings_by_neighborhood)

					if poi_state == source_state:
						MarkKey(source_state, poi_type, (source_id, poi_id), (source_ping_count * median_dwell), visits)
	return pings_by_neighborhood, visits

def BuildPoiSerialization(pings_for_state, visits_for_state, census_block_stats):
	id_to_neighborhood = []
	neighborhood_to_id = {}
	def GetId(neighborhood):
		if neighborhood not in neighborhood_to_id:
			new_id = len(neighborhood_to_id)
			neighborhood_to_id[neighborhood] = new_id
			id_to_neighborhood.append(neighborhood)
		return neighborhood_to_id[neighborhood]

	poi_to_id = {}
	def GetPoiId(poi):
		if poi not in poi_to_id:
			new_id = len(poi_to_id)
			poi_to_id[poi] = new_id
		return poi_to_id[poi]

	print("Loading {} neighborhoods".format(len(pings_for_state)))
	print("Loading {} neighborhood connections".format(len(visits_for_state)))
	
	for neighborhood in pings_for_state.keys():
		if neighborhood not in neighborhood_to_id:
			GetId(neighborhood)
	for (home_neighborhood, visit_poi) in visits_for_state.keys():
		GetId(home_neighborhood)
		GetPoiId(visit_poi)

	print("Computing weighted totals")
	print("Matrix size #{}x{}".format(len(neighborhood_to_id), len(poi_to_id)))
	data = []
	rows = []
	cols = []
	for (home_neighborhood, visit_poi), count in visits_for_state.iteritems():
		data.append(count)
		rows.append(GetId(home_neighborhood))
		cols.append(GetPoiId(visit_poi))
	sparse_matrix = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(len(neighborhood_to_id), len(poi_to_id)))
	interconnect = sparse_matrix * sparse_matrix.transpose()


	print("Computed interconnect")
	coo_matrix = interconnect.tocoo()
	print("Converted matrix")

	neighborhood_stats = collections.defaultdict(NeighborhoodStats)

	print("Loading {} connections".format(len(coo_matrix.row)))
	for i in range(len(coo_matrix.row)):
		if i % 1000000 == 0:
			print("Loading connection {}".format(i))
		source = id_to_neighborhood[coo_matrix.row[i]]
		dest = id_to_neighborhood[coo_matrix.col[i]]
		neighborhood_stats[source].AddInteractions(dest, coo_matrix.data[i])


	print("Finalizing stats")
	for neighborhood, count in pings_for_state.iteritems():
		population = census_block_stats[neighborhood]['population']
		pings_per_pop = count / (population + 0.0)
		neighborhood_stats[neighborhood].total_pings = pings_per_pop
		neighborhood_stats[neighborhood].population = population
		neighborhood_stats[neighborhood].ReduceNeighbors(.001)


	return neighborhood_stats	



def ConvertToInterconnectData(state_stats):
	interconnect_data_map = {}
	for category, category_stats in state_stats.iteritems():
		for neighborhood, neighborhood_stats in category_stats.iteritems():
			category_data = data_parser.CategoryData(neighborhood_stats.total_pings, neighborhood_stats.neighbors)
			if neighborhood not in interconnect_data_map:
				interconnect_data_map[neighborhood] = data_parser.InterconnectData(neighborhood, neighborhood_stats.population, {})
			interconnect_data = interconnect_data_map[neighborhood]
			interconnect_data.category_map[category] = category_data
	return interconnect_data_map


def GetPerCategoryStats(poi_pattern_file_names, poi_to_category, row_key_extractor=ConvertRowKey):
	per_category_count = {}
	for file_name in poi_pattern_file_names:
		print("Parsing file: {}".format(file_name))
		with gzip.open(file_name) as f:
			csv_dict_reader = csv.DictReader(f)
			for i, row in enumerate(csv_dict_reader):
				if i % 100000 == 0:
					print("Loading row #{}: POI:{}".format(i, row['location_name']))
				row_key = row_key_extractor(row['poi_cbg'])
				poi_type = poi_to_category.get(row['safegraph_place_id'], None)
				if poi_type:
					combined_key = (row_key, poi_type)
					per_day_visits = json.loads(row['visits_by_day'])
					existing = per_category_count.get(combined_key)
					if not existing:
						per_category_count[combined_key] = per_day_visits
					else:
						new_data = [per_day_visits[i] + existing[i] for i in range(len(existing))]
						per_category_count[combined_key] = new_data
	return per_category_count



def SplitIntoSerializedPoiStates(poi_pattern_file_names, census_block_stats_file_name, poi_category_file_names, row_key_extractor=ConvertRowKey, filter_states=None):
	print("Joining Census Data")
	census_block_stats = GetCensusBlockStats(census_block_stats_file_name, row_key_extractor)

	print("Loading Category Names")
	poi_category_map = MapPoiToCategory(poi_category_file_names)

	print("Loading Neighborhoods")
	pings, visits = LoadPoiPatterns(poi_pattern_file_names, census_block_stats, poi_category_map, row_key_extractor, filter_states)
	combined_data_by_state = {}
	for state in pings.keys():
		print("Loading state: {} ".format(state))
		if state is None:
			continue
		state_stats = {}
		for category in pings[state].keys():
			print("Loading Category: {}".format(category))
			state_stats[category] = BuildPoiSerialization(pings.get(state, {}).get(category, {}), visits.get(state, {}).get(category, {}), census_block_stats)
		print("Serializing")
		serialized = ConvertToInterconnectData(state_stats)
		combined_data_by_state[state] = serialized
		
	return combined_data_by_state






