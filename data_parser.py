import csv
import pickle
import datetime
import gzip
import marshal
import os
import errno
import data_parser

def LoadPickle(file_name):
	if file_name.endswith(".gz"):
		with gzip.open(file_name) as f:
			return pickle.load(f)
	else:
		with open(file_name) as f:
			return pickle.load(f)




def ConvertCensusData(census_data_file):
	pop_by_county = {}
	with open(census_data_file) as f:
		dict_reader = csv.DictReader(f)
		for line in dict_reader:
			if line['SUMLEV'] == '050':
				fips = line['STATE'] + line['COUNTY']
				pop = float(line['POPESTIMATE2019'])
				pop_by_county[fips] = pop
	return pop_by_county

def LoadPopulationByCounty(population_pickle):
	with open(population_pickle) as f:
		return pickle.load(f)	

class CountyEstimate():
	def __init__(self, fips, starting_date):
		self.fips = fips
		self.starting_date = starting_date
		self.data_by_date = []
	def AddDay(self, new_cum):
		old_total = 0 if not self.data_by_date else self.data_by_date[-1][0]
		diff = new_cum - old_total
		self.data_by_date.append((new_cum, diff))


def ExtractStats(county_estimate, target_date, num_prev_days):
	date_diff = (target_date - county_estimate.starting_date).days
	target_finish_index = date_diff
	curr_infections = [0] * num_prev_days

	def GetEstimate(index):
		if index < 0:
			return (0, 0)
		if index >= len(county_estimate.data_by_date):
			return (0, 0)
		return county_estimate.data_by_date[index]

	ever_sick = GetEstimate(target_finish_index)[0]
	for i in range(num_prev_days):
		target_index = target_finish_index - i
		curr_infections[i] = GetEstimate(target_index)[1]
	return (ever_sick, curr_infections)

def ConvertEstimates(estimate_file):
	estimate_by_county = {}
	with open(estimate_file) as f:
		dict_reader = csv.DictReader(f)
		curr_county = None

		for line in dict_reader:
			fips = line['fips']
			if curr_county is None:
				curr_county = CountyEstimate(fips, datetime.datetime.strptime(line['date'], "%Y-%m-%d"))
			elif fips != curr_county.fips:
				estimate_by_county[curr_county.fips] = curr_county
				curr_county = CountyEstimate(fips, datetime.datetime.strptime(line['date'], "%Y-%m-%d"))
			curr_county.AddDay(float(line['cum.incidence']))

		estimate_by_county[curr_county.fips] = curr_county 
	return estimate_by_county


# Data classes for storing the interconnect data, this is computed in safegraph parser.
class CategoryData():
	def __init__(self, transmission, neighbors):
		self.transmission = transmission
		self.neighbors = neighbors

class InterconnectData():
	def __init__(self, block, normalization_factor, category_map):
		self.block = block
		self.normalization_factor = normalization_factor
		self.category_map = category_map


CATEGORIES = ['EDUCATION', 'HEALTH_CARE', 'OTHER', 'RELIGION', 'RESTAURANT']

def EmptyInterconnect(block):
	cat_map = {cat: CategoryData(0, {}) for cat in CATEGORIES}
	return InterconnectData(block, 1, cat_map)

def CombineMonthlyInterconnect(interconnect_by_month):
	all_blocks = set()
	for month, interconnect in interconnect_by_month.iteritems():
		for block in interconnect.keys():
			all_blocks.add(block)
	all_sorted_blocks = sorted(all_blocks)

	monthly_sorted_values = {}
	for month, interconnect in interconnect_by_month.iteritems():
		monthly_sorted_values[month] = [interconnect.get(block, EmptyInterconnect(block)) for block in all_sorted_blocks]
	return monthly_sorted_values

def StoreMarshalInterconnect(interconnect_data, marshal_file):
	def MakeMarshal(interconnect):
		cat_map = {}
		for cat, cat_data in interconnect.category_map.iteritems():
			cat_map[cat] = (cat_data.transmission, {k: float(v) for k, v in cat_data.neighbors.iteritems()})
		return (interconnect.normalization_factor, cat_map)
	marshalable_data = {k: MakeMarshal(interconnect) for k, interconnect in interconnect_data.iteritems()}
	if not os.path.exists(os.path.dirname(marshal_file)):
	    try:
	        os.makedirs(os.path.dirname(marshal_file))
	    except OSError as exc: # Guard against race condition
	        if exc.errno != errno.EEXIST:
	            raise

	with open(marshal_file, 'w') as g:
		marshal.dump(marshalable_data, g)

def LoadMarshalInterconnect(marshal_file):
	interconnect_map = {}
	with open(marshal_file) as f:
		marshalled_data = marshal.load(f)
	for key, (normalization_factor, cat_map) in marshalled_data.iteritems():
		cat_data_map = {cat: CategoryData(transmission, neighbors) for cat, (transmission, neighbors) in cat_map.iteritems()}
		interconnect_map[key] = InterconnectData(key, normalization_factor, cat_data_map)
	return interconnect_map


def LoadMonthlyMarshalInterconnect(marshal_file_by_month):
	raw_monthly_interconnect = {month : LoadMarshalInterconnect(marshal_file) for month, marshal_file in marshal_file_by_month.iteritems()}
	return CombineMonthlyInterconnect(raw_monthly_interconnect)

def LoadRawDiseaseStats(disease_pickle):
	with open(disease_pickle) as f:
		raw_disease_by_county = pickle.load(f)
	return raw_disease_by_county


def LoadDiseaseStatsByCounty(disease_stats, load_date, infection_duration):

	disease_by_county = {}
	for county, county_stats in disease_stats.iteritems():
		ever_sick, current_infections = ExtractStats(county_stats, load_date, infection_duration)
		disease_by_county[county] = (ever_sick, current_infections)
	return disease_by_county