from __future__ import division
import numpy as np
import scipy.optimize
import collections
import csv
import copy
import random
import heapq
import data_parser
import datetime
import pickle
import scipy.sparse

INFECTION_DURATION = 7

class BlockStats():
	def __init__(self, block_id, population, currently_infected, ever_infected):
		self.block_id = block_id
		# Population for the block
		self.population = population
		# Start off assuming that those currently infected are evenly split between their days of infection.
		self.current_infected_by_day = collections.deque()
		self.current_infected = 0
		for current_infected in currently_infected:
			self.current_infected_by_day.appendleft(current_infected)
			self.current_infected += current_infected
		# Start off with some proportion already immune due to already being infected.
		self.ever_infected = ever_infected

	def __repr__(self):
		return "Block:{} Population:{} Current Infected:{:.2%} Ever Infected: {:.2%}".format(
			self.block_id, self.population, self.current_infected / self.population, 
			self.ever_infected / self.population)

	def AdvanceDay(self, new_infections):
		# Update the total infected.
		self.current_infected += new_infections
		self.current_infected_by_day.appendleft(new_infections)
		recovered = self.current_infected_by_day.pop()
		self.current_infected -= recovered
		self.ever_infected += new_infections

	def ComputeActualInfections(self, raw_infections):
		# Probability that a single infection transmission happens to an individual.
		individual_infection = 1 / self.population

		# Probability that a single individual is not infected by any of the infections.
		individual_miss_infection = (1 - individual_infection) ** raw_infections

		# The average infected is the sum for each non infected person of getting sick.
		average_infected = (1 - individual_miss_infection) * (self.population - self.ever_infected)
		return average_infected

def CopyBlock(block):
	current_infected_copy = [n for n in block.current_infected_by_day]
	current_infected_copy.reverse()

	return BlockStats(block.block_id, block.population, current_infected_copy, block.ever_infected)


class CategoryData():
	def __init__(self, transmission, neighbors):
		self.transmission = transmission
		self.neighbors = neighbors

class InterconnectData():
	def __init__(self, block, normalization_factor, category_map):
		self.block = block
		self.normalization_factor = normalization_factor
		self.category_map = category_map


def GetUnweightedTransmissionStatsByCategory(raw_interconnect_data, category):
	index_for_block = {}
	block_ids = []
	transmission_matrix = np.zeros((len(raw_interconnect_data), len(raw_interconnect_data)))
	transmission_per_block = np.zeros((len(raw_interconnect_data)))

	for i,raw_data in enumerate(raw_interconnect_data):
		index_for_block[raw_data.block] = i 
		block_ids.append(raw_data.block)

	for i, raw_data in enumerate(raw_interconnect_data):
		if category in raw_data.category_map:
			raw_category_data = raw_data.category_map[category]
			transmission_per_block[i] = raw_category_data.transmission
			for neighbor, transmission_fraction in raw_category_data.neighbors.iteritems():
				transmission_matrix[i][index_for_block[neighbor]] += transmission_fraction
	return (transmission_per_block, transmission_matrix)


class TransmissionStatistics():
	def __init__(self, beta_by_category, internal_transmission, raw_interconnect_data, unweighted_transmission_stats):

		self.internal_transmission = internal_transmission / INFECTION_DURATION
		
		self.total_transmission_by_block = np.zeros((len(raw_interconnect_data),))
		self.index_for_block = {}
		self.block_ids = []

		temp_transmission_matrix = np.zeros((len(raw_interconnect_data), len(raw_interconnect_data)))

		for i,raw_data in enumerate(raw_interconnect_data):
			self.index_for_block[raw_data.block] = i 
			self.block_ids.append(raw_data.block)

		for category, (transmission_per_block, transmission_matrix) in unweighted_transmission_stats.iteritems():
			self.total_transmission_by_block += (transmission_per_block * beta_by_category.get(category, 0.0) / INFECTION_DURATION)
			temp_transmission_matrix += transmission_matrix * beta_by_category.get(category, 0.0)

		for i in range(len(temp_transmission_matrix)):
			row_sum = np.sum(temp_transmission_matrix[i])
			if row_sum != 0:
				temp_transmission_matrix[i] /= row_sum

		self.transmission_matrix = temp_transmission_matrix



class InternalState():
	def __init__(self, size):
		self.block_combined_transmission = np.zeros([size])
		self.internal_transmission = np.zeros([size])
		self.current_infected = np.zeros([size])


def AdvanceDay(block_stats, transmission_stats, internal_state):

	for i, block in enumerate(block_stats):
		block_index = i
		internal_state.current_infected[i] = block.current_infected

	internal_state.block_combined_transmission = internal_state.current_infected * transmission_stats.total_transmission_by_block
	internal_state.internal_transmission = internal_state.current_infected * transmission_stats.internal_transmission
	received_infections = np.matmul(internal_state.block_combined_transmission, transmission_stats.transmission_matrix)

	for i, block in enumerate(block_stats):
		new_infections = received_infections[i] + internal_state.internal_transmission[i]
		block.AdvanceDay(block.ComputeActualInfections(new_infections))


def FixFipsCode(row_code):
	# Excel sometimes strips leading zeros. Fix this when it occurs.
	if len(row_code) == 4:
		return '0' + row_code
	else:
		return row_code

def LoadDiseaseStatsByCounty(disease_pickle, month):
	month_start = datetime.datetime(2020, month, 1)
	with open(disease_pickle) as f:
		raw_disease_by_county = pickle.load(f)

	disease_by_county = {}
	for county, county_stats in raw_disease_by_county.iteritems():
		ever_sick, current_infections = data_parser.ExtractStats(county_stats, month_start, INFECTION_DURATION)
		disease_by_county[county] = (ever_sick, current_infections)
	return disease_by_county



def LoadPopulationByCounty(population_pickle):
	with open(population_pickle) as f:
		return pickle.load(f)

def GetCounty(block_id):
	return block_id[:5]

def LoadInitialState(interconnect_data, disease_by_county, population_by_county):
	total_normalization_by_county = collections.defaultdict(int)


	for interconnect in interconnect_data:
		total_normalization_by_county[GetCounty(interconnect.block)] += interconnect.normalization_factor


	blocks = []
	for interconnect in interconnect_data:
		county = GetCounty(interconnect.block)
		ever_sick, current_sick = disease_by_county.get(county, (0, [0] * INFECTION_DURATION))
		population = population_by_county.get(county, 1)

		total_normalization_factor = total_normalization_by_county[county]

		fractional_part = interconnect.normalization_factor / total_normalization_factor

		blocks.append(BlockStats(interconnect.block, population * fractional_part, [c * fractional_part for c in current_sick], ever_sick * fractional_part))
	return blocks

def ComputeProgression(original_blocks, transmission_stats, simulation_days):
	copied =  [CopyBlock(block) for block in original_blocks]
	internal_state = InternalState(len(copied))
	for day in range(simulation_days):
		AdvanceDay(copied, transmission_stats, internal_state)
	return copied

def ComputeWeightedError(predicted_blocks, final_disease_by_county):
	total_weight = 0
	total_weighted_error = 0
	pop_by_county = collections.defaultdict(float)
	predicted_ever_sick_by_county = collections.defaultdict(float)
	for block in predicted_blocks:
		county = GetCounty(block.block_id)
		pop_by_county[county] += block.population
		predicted_ever_sick_by_county[county] += block.ever_infected
	for county, predicted in predicted_ever_sick_by_county.iteritems():
		actual = final_disease_by_county.get(county, (0, 0))[0]
		relative_error = (predicted - actual) / (max(predicted, actual))
		square_error = relative_error * relative_error
		actual_pop = pop_by_county.get(county, 0)
		total_weight += actual_pop
		total_weighted_error += actual_pop * square_error

	return (total_weighted_error / total_weight)

def AssessParameters(starting_blocks, final_disease_by_county, simulation_days, beta_by_category, internal_transmission, interconnect, unweighted_transmission_stats):
	transmission_stats = TransmissionStatistics(beta_by_category, internal_transmission, interconnect, unweighted_transmission_stats)
	predicted_final_blocks = ComputeProgression(starting_blocks, transmission_stats, simulation_days)
	prediction_error = ComputeWeightedError(predicted_final_blocks, final_disease_by_county)

	return predicted_final_blocks, prediction_error

CATEGORIES = ['OTHER', 'RESTAURANT', 'EDUCATION', 'RELIGION', 'HEALTH_CARE']

def GetStructured(random_params):
	internal_transmission = abs(random_params[0])
	cat_map  = {}
	for i, category in enumerate(CATEGORIES):
		cat_map[category] = abs(random_params[i+1])
	return internal_transmission, cat_map

class RandomParamRanges():
	def __init__(self, internal_range, range_by_cat):
		self.internal_range = internal_range
		self.range_by_cat = range_by_cat

	def GetRandom(self):
		random_params = [random.uniform(self.internal_range[0], self.internal_range[1])]
		for category in CATEGORIES:
			min_val, max_val = self.range_by_cat[category]
			random_params.append(random.uniform(min_val, max_val))
		return random_params

class UniformRisk():
	def __init__(self, internal_range, cat_range):
		self.internal_range = internal_range
		self.cat_range = cat_range

	def GetRandom(self):
		random_params = [random.uniform(self.internal_range[0], self.internal_range[1])]
		cat_val = random.uniform(self.cat_range[0], self.cat_range[1])
		for category in CATEGORIES:
			random_params.append(cat_val)
		return random_params		

def DefaultRandomParams():
	range_by_cat = {}
	for category in CATEGORIES:
		range_by_cat[category] = (0, .02)
	return RandomParamRanges((0, 2), range_by_cat)

def ComputeRandomParameters(population_pickle, disease_pickle, starting_month, ending_month, interconnect, num_attempts=10000, num_to_keep=10, random_ranges=None):
	if random_ranges is None:
		random_ranges = DefaultRandomParams()

	population_by_county = LoadPopulationByCounty(population_pickle)
	starting_disease_by_county = LoadDiseaseStatsByCounty(disease_pickle, starting_month)
	final_disease_by_county = LoadDiseaseStatsByCounty(disease_pickle, ending_month)

	starting_blocks = LoadInitialState(interconnect, starting_disease_by_county, population_by_county)

	unweighted_transmission_stats = {}
	for category in CATEGORIES:
		unweighted_transmission_stats[category] = GetUnweightedTransmissionStatsByCategory(interconnect, category)


	def ComputeRandomAttempt(random_params):
		internal_transmission, cat_map = GetStructured(random_params)
		predicted_final_blocks, error = AssessParameters(
			starting_blocks, final_disease_by_county, 
			(ending_month - starting_month) * 30, 
			cat_map, 
			internal_transmission, 
			interconnect, 
			unweighted_transmission_stats)
		return error

	best_results = []
	for i in range(num_attempts):
		if (i % 2) == 1 and best_results:
			best_result_index = random.randint(0, len(best_results) - 1)
			best_result_params = best_results[best_result_index][1]
			random_params = [param * random.uniform(.95, 1.05) for param in best_result_params]
		else:
			random_params = random_ranges.GetRandom()
		error = ComputeRandomAttempt(random_params)
		if len(best_results) < num_to_keep:
			heapq.heappush(best_results, (-error, random_params))
		else:
			heapq.heappushpop(best_results, (-error, random_params))
		if (i % 10) == 0:
			best_result = max(best_results)
			print("Attempt: #{} Best Params:{} Best Error:{}".format(i, GetStructured(best_result[1]), -best_result[0]))			
	return sorted(best_results, key=lambda x: -x[0])

	# starting_guess = [0.0] + [0.00001] * len(CATEGORIES)
	# return scipy.optimize.basinhopping(ComputeRandomAttempt, starting_guess, niter=num_attempts, T=.001, stepsize=.001, disp=True)

def GetPrediction(random_params, population_pickle, disease_pickle, starting_month, ending_month, interconnect):
	population_by_county = LoadPopulationByCounty(population_pickle)
	starting_disease_by_county = LoadDiseaseStatsByCounty(disease_pickle, starting_month)
	final_disease_by_county = LoadDiseaseStatsByCounty(disease_pickle, ending_month)

	starting_blocks = LoadInitialState(interconnect, starting_disease_by_county, population_by_county)

	unweighted_transmission_stats = {}
	for category in CATEGORIES:
		unweighted_transmission_stats[category] = GetUnweightedTransmissionStatsByCategory(interconnect, category)
	internal, cat_map = GetStructured(random_params)
	prediction, error =  AssessParameters(starting_blocks, final_disease_by_county, 
			(ending_month - starting_month) * 30, 
			cat_map, 
			internal, 
			interconnect, 
			unweighted_transmission_stats)
	return starting_blocks, prediction, error










