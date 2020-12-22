import csv
import pickle
import datetime

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
