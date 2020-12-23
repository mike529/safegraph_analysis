from __future__ import division
import numpy as np
import scipy.optimize
import collections
import csv
import random
import heapq
import data_parser
import datetime
import pickle
import scipy.sparse

INFECTION_DURATION = 7

CATEGORIES = ['EDUCATION', 'HEALTH_CARE', 'OTHER', 'RELIGION', 'RESTAURANT']

# Stores the internal state for a given block of people.
# Within the block, infection is assumed to be uniform and homogenous.
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

# Fast copying of blocks. Significantly faster than copy.deepcopy
def CopyBlock(block):
	current_infected_copy = [n for n in block.current_infected_by_day]
	current_infected_copy.reverse()

	return BlockStats(block.block_id, block.population, current_infected_copy, block.ever_infected)

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


# Compute the internal matrices for transmission by each category.
# When training the data we need to mutate the weights frequently and it is much quicker to
# remultiply the weights than to recompute from scratch.
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
	def __init__(self, transmission_per_block, transmission_matrix, internal_transmission=0, homogenous_transmission=0):
		self.transmission_per_block = transmission_per_block
		self.transmission_matrix = transmission_matrix
		self.internal_transmission = internal_transmission
		self.homogenous_transmission = homogenous_transmission

class StatsContainer():
	def __init__(self, raw_interconnect_data):
		self.block_length = len(raw_interconnect_data)
		self.cat_map = {}
		for category in CATEGORIES:
			self.cat_map[category] = GetUnweightedTransmissionStatsByCategory(raw_interconnect_data, category)
	
	def GetWeightedStats(self, beta_by_category, internal_transmission=0, homogenous_transmission=0):
		
		total_transmission_by_block = np.zeros((self.block_length),)
		temp_transmission_matrix = np.zeros((self.block_length, self.block_length))

		for category, (transmission_per_block, transmission_matrix) in self.cat_map.iteritems():
			total_transmission_by_block += (transmission_per_block * beta_by_category.get(category, 0.0))
			temp_transmission_matrix += transmission_matrix * beta_by_category.get(category, 0.0)

		for i in range(len(temp_transmission_matrix)):
			row_sum = np.sum(temp_transmission_matrix[i])
			if row_sum != 0:
				temp_transmission_matrix[i] /= row_sum

		return TransmissionStatistics(total_transmission_by_block, temp_transmission_matrix, internal_transmission, homogenous_transmission)		


def AdvanceDay(block_stats, transmission_stats):

	current_infected = np.zeros(len(block_stats,))
	block_pops = np.zeros(len(block_stats,))
	total_pop = 0
	for i, block in enumerate(block_stats):
		block_index = i
		current_infected[i] = block.current_infected
		block_pops[i] = block.population
	total_pop = np.sum(block_pops)
	total_infected = np.sum(current_infected)
	homogenous_transmission = block_pops * total_infected * transmission_stats.homogenous_transmission / total_pop

	block_combined_transmission = current_infected * transmission_stats.transmission_per_block
	internal_transmission = current_infected * transmission_stats.internal_transmission
	received_infections = np.matmul(block_combined_transmission, transmission_stats.transmission_matrix)

	new_infections = (received_infections + internal_transmission + homogenous_transmission) / INFECTION_DURATION
	for i, block in enumerate(block_stats):
		block.AdvanceDay(block.ComputeActualInfections(new_infections[i]))

def ComputeProgression(blocks, transmission_stats, simulation_days):
	copied = [CopyBlock(block) for block in blocks]
	for i in range(simulation_days):
		AdvanceDay(copied, transmission_stats)
	return copied

def GetCounty(block_id):
	return block_id[:5]

# Computes the initial stats, this is complicated by trying to associate diseases to census tracts rather than counties.
# Right now we use a uniform prior that within a county at the start disease is uniformly spread according to the initial population.
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


# Computes county level stats for a set of blocks.
def GetStatsByCounty(blocks):
	pop_by_county = collections.defaultdict(float)
	predicted_ever_sick_by_county = collections.defaultdict(float)
	current_sick_by_county = collections.defaultdict(float)
	for block in blocks:
		county = GetCounty(block.block_id)
		pop_by_county[county] += block.population
		predicted_ever_sick_by_county[county] += block.ever_infected
		current_sick_by_county[county] += block.current_infected

	combined_data = {}
	for county, pop in pop_by_county.iteritems():
		combined_data[county] = (pop, current_sick_by_county[county], predicted_ever_sick_by_county[county])

	return combined_data


# Given a set of predicted blocks, along with observations by county, compute an error metric.
# We do this by computing for each county the percentage error on the number of ever infected.
# This is then reweighted by population.
def ComputeWeightedError(predicted_blocks, disease_stats, current_date):
	total_weight = 0
	total_weighted_error = 0
	stats_by_county = GetStatsByCounty(predicted_blocks)
	def GetRelativeError(predicted, actual):
		if predicted == 0 and actual == 0:
			return 0
		else:
			return ((predicted - actual) / actual) ** 2

	for county, (pop, predicted_current_sick, predicted_ever_sick) in stats_by_county.iteritems():
		actual_ever, actual_current = data_parser.ExtractStats(disease_stats[county], current_date, INFECTION_DURATION)
		relative_error = GetRelativeError(predicted_ever_sick + 10, actual_ever + 10)
		relative_error += GetRelativeError(predicted_current_sick + 10, sum(actual_current) + 10)
		total_weight += (pop * 2) 
		total_weighted_error += pop * relative_error

	return (total_weighted_error / total_weight)


class TransmissionParameters():
	def __init__(self, beta_by_category, internal_transmission, homogenous_transmission, rounding=None):
		self.beta_by_category = beta_by_category
		self.internal_transmission = internal_transmission
		self.homogenous_transmission = homogenous_transmission
		if rounding:
			self.homogenous_transmission = round(self.homogenous_transmission, rounding)
			self.internal_transmission = round(self.internal_transmission, rounding)
			for cat in self.beta_by_category:
				self.beta_by_category[cat] = round(self.beta_by_category[cat], rounding)
	def __repr__(self):
		return "Internal: {} Homogenous: {} Activity Based: {}".format(self.internal_transmission, self.homogenous_transmission, self.beta_by_category)

def AssessParameters(starting_blocks, disease_stats, start_date, end_date, transmission_parameters, stats_container_by_month, evaluation_intervals, include_interval_blocks):
	transmission_stats_by_month = {
		month: stats_container.GetWeightedStats(
			transmission_parameters.beta_by_category, 
			transmission_parameters.internal_transmission, 
			transmission_parameters.homogenous_transmission) for month, stats_container in stats_container_by_month.iteritems()
		}
	current_date = start_date
	copied_blocks = [CopyBlock(block) for block in starting_blocks]

	prediction_errors = []
	blocks_by_interval = {}
	if include_interval_blocks:
		blocks_by_interval[current_date] = [CopyBlock(block) for block in copied_blocks]
	while current_date < end_date:
		AdvanceDay(copied_blocks, transmission_stats_by_month[current_date.month])
		current_date += datetime.timedelta(days=1)
		if current_date in evaluation_intervals or current_date == end_date:
			prediction_errors.append(ComputeWeightedError(copied_blocks, disease_stats, current_date))
			if include_interval_blocks:
				blocks_by_interval[current_date] = [CopyBlock(block) for block in  copied_blocks]


	return blocks_by_interval, (sum(prediction_errors) / len(prediction_errors))

class UniformRandomParams():
	def __init__(self, internal_range, homogenous_range, cat_range):
		self.internal_range = internal_range
		self.homogenous_range = homogenous_range
		self.cat_range = cat_range

	def GetNewRandom(self):
		internal_transmission = random.uniform(*self.internal_range)
		homogenous_transmission = random.uniform(*self.homogenous_range)
		cat_map = {cat: random.uniform(*self.cat_range) for cat in CATEGORIES}
		return TransmissionParameters(cat_map, internal_transmission, homogenous_transmission, rounding=4)

	def MutateRandom(self, transmission_parameters):
		internal_transmission = random.uniform(.95, 1.05) * transmission_parameters.internal_transmission
		homogenous_transmission = random.uniform(.95, 1.05) * transmission_parameters.homogenous_transmission
		cat_map = {cat:random.uniform(.95, 1.05) * t for cat, t in transmission_parameters.beta_by_category.iteritems()}
		return TransmissionParameters(cat_map, internal_transmission, homogenous_transmission, rounding=4)


def DefaultRandomParams():
	return UniformRandomParams((0,2.0), (0, 2.0), (0, .005))


def BuildAssessmentFunction(population_by_county, disease_stats, starting_date, ending_date, interconnect_by_month, evaluation_intervals=None, include_interval_blocks=False):
	if evaluation_intervals is None:
		evaluation_intervals = set(datetime.datetime(2020, month, 1) for month in range(starting_date.month + 1, ending_date.month + 1))


	starting_disease_by_county = data_parser.LoadDiseaseStatsByCounty(disease_stats, starting_date, INFECTION_DURATION)

	starting_blocks = LoadInitialState(interconnect_by_month[starting_date.month], starting_disease_by_county, population_by_county)
	starting_stats = GetStatsByCounty(starting_blocks)
	stats_container_by_month = {}
	for month, interconnect in interconnect_by_month.iteritems():
		stats_container_by_month[month] = StatsContainer(interconnect)

	def ComputeAttempt(transmission_parameters):
		return AssessParameters(starting_blocks, disease_stats, starting_date, ending_date, transmission_parameters, stats_container_by_month, evaluation_intervals, include_interval_blocks)
	return ComputeAttempt

def ComputeRandomParameters(population_by_county, disease_stats, starting_date, ending_date, interconnect_by_month, num_attempts=2000, num_to_keep=10, random_param_gen=None, evaluation_intervals=None):
	if random_param_gen is None:
		random_param_gen  = DefaultRandomParams()

	compute_attempt = BuildAssessmentFunction(population_by_county, disease_stats, starting_date, ending_date, interconnect_by_month, evaluation_intervals, include_interval_blocks=False)

	best_results = []
	for i in range(num_attempts):
		if (i % 2) == 1 and best_results:
			best_result_index = random.randint(0, len(best_results) - 1)
			best_result_params = best_results[best_result_index][1]

			random_params = random_param_gen.MutateRandom(best_result_params)
		else:
			random_params = random_param_gen.GetNewRandom()
		_,_, error = compute_attempt(random_params)
		if len(best_results) < num_to_keep:
			heapq.heappush(best_results, (-error, random_params))
		else:
			heapq.heappushpop(best_results, (-error, random_params))
		if (i % 10) == 0:
			best_result = max(best_results)
			print("Attempt: #{} Best Params:{} Best Error:{}".format(i, best_result[1], -best_result[0]))			
	return sorted(best_results, key=lambda x: -x[0])


def RunScenario(population_by_county, disease_stats, starting_date, ending_date, interconnect_by_month, transmission_parameters, evaluation_intervals=None):
	compute_attempt = BuildAssessmentFunction(population_by_county, disease_stats, starting_date, ending_date, interconnect_by_month, evaluation_intervals, include_interval_blocks=True)

	computed_blocks, error = compute_attempt(transmission_parameters)
	predicted_final_blocks = computed_blocks[ending_date]
	predicted_stats_by_county = GetStatsByCounty(predicted_final_blocks)
	
	starting_disease_by_county = data_parser.LoadDiseaseStatsByCounty(disease_stats, starting_date, INFECTION_DURATION)
	final_disease_by_county = data_parser.LoadDiseaseStatsByCounty(disease_stats, ending_date, INFECTION_DURATION)

	combined_output = {}
	for county, (pop, predicted_current_sick, predicted_ever_sick) in predicted_stats_by_county.iteritems():
		found_actual = final_disease_by_county.get(county, (0, []))
		found_starting = starting_disease_by_county.get(county, (0, []))
		combined_output[county] = {
			'pop': pop, 
			'ever_sick': {
				'start': found_starting[0],
				'finish': found_actual[0],
				'predicted': predicted_ever_sick
			},
			'current_sick': {
				'start': sum(found_starting[1]),
				'finish': sum(found_actual[1]),
				'predicted': predicted_current_sick
			}
		}

	return computed_blocks, combined_output, error


	









