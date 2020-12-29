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
import sklearn.svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
		self.current_infected_by_day = []
		self.current_infected = 0
		for current_infected in currently_infected:
			self.current_infected_by_day.append(current_infected)
			self.current_infected += current_infected
		self.current_index = len(self.current_infected_by_day) - 1
		# Start off with some proportion already immune due to already being infected.
		self.ever_infected = ever_infected

	def __repr__(self):
		return "Block:{} Population:{} Current Infected:{:.2%} Ever Infected: {:.2%}".format(
			self.block_id, self.population, self.current_infected / self.population, 
			self.ever_infected / self.population)

	def AdvanceDay(self, new_infections):
		# Update the total infected.
		self.current_infected += new_infections
		recovered = self.current_infected_by_day[self.current_index]
		self.current_infected -= recovered
		self.current_infected_by_day[self.current_index] = new_infections
		self.current_index = (self.current_index + 1) % len(self.current_infected_by_day)
		self.ever_infected += new_infections

	def ComputeActualInfections(self, raw_infections):
		# Probability that a single infection transmission happens to an individual.
		individual_infection = 1 / self.population

		# Probability that a single individual is not infected by any of the infections.
		individual_miss_infection = (1 - individual_infection) ** raw_infections

		# The average infected is the sum for each non infected person of getting sick.
		average_infected = (1 - individual_miss_infection) * (self.population - self.ever_infected)
		return average_infected

# class SubBlockParams():
# 	def __init__(self, population, transmission_weight):
# 		self.population = population
# 		self.transmission_weight = transmission_weight

# class GroupedBlock():
# 	def __init__(block_id, sub_blocks, currently_infected, ever_infected):
# 		self.total_population = sum(sub_block.population for sub_block in sub_blocks)
# 		self.total_weighted_transmission = sum(sub_block.population * sub_block.transmission_weight)


# 		self.blocks = [BlockStats(block_id, sub_block.population, sub_block.)]


# Fast copying of blocks. Significantly faster than copy.deepcopy
def CopyBlock(block):
	current_infected_copy = [n for n in block.current_infected_by_day]
	current_infected_copy.reverse()

	return BlockStats(block.block_id, block.population, current_infected_copy, block.ever_infected)

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
		self.index_for_block = {}
		self.cat_map = {}
		for category in CATEGORIES:
			self.cat_map[category] = GetUnweightedTransmissionStatsByCategory(raw_interconnect_data, category)
	
	def GetWeightedStats(self, beta_by_category, internal_transmission=0, homogenous_transmission=0, monthly_factor=1.0):
		
		total_transmission_by_block = np.zeros((self.block_length),)
		temp_transmission_matrix = np.zeros((self.block_length, self.block_length))

		for category, (transmission_per_block, transmission_matrix) in self.cat_map.iteritems():
			total_transmission_by_block += (transmission_per_block * beta_by_category.get(category, 0.0) * monthly_factor)
			temp_transmission_matrix += transmission_matrix * beta_by_category.get(category, 0.0)

		for i in range(len(temp_transmission_matrix)):
			row_sum = np.sum(temp_transmission_matrix[i])
			if row_sum != 0:
				temp_transmission_matrix[i] /= row_sum

		return TransmissionStatistics(total_transmission_by_block, temp_transmission_matrix, internal_transmission * monthly_factor, homogenous_transmission * monthly_factor)		


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
			return (2 *  abs(predicted - actual)) / (predicted + actual)
	total_predicted = 0
	total_actual = 0
	total_pop = 0

	for county, (pop, predicted_current_sick, predicted_ever_sick) in stats_by_county.iteritems():
		actual_ever, actual_current = data_parser.ExtractStats(disease_stats[county], current_date, INFECTION_DURATION)
		relative_error = GetRelativeError(predicted_ever_sick, actual_ever)
		total_weight += pop 
		total_weighted_error += pop * relative_error
		total_predicted += predicted_ever_sick
		total_actual += actual_ever
		
	return total_weighted_error / total_weight


class TransmissionParameters():
	def __init__(self, beta_by_category, internal_transmission, homogenous_transmission, monthly_factor, rounding=None):
		self.beta_by_category = beta_by_category
		self.internal_transmission = internal_transmission
		self.homogenous_transmission = homogenous_transmission
		self.monthly_factor = monthly_factor
		if rounding:
			self.homogenous_transmission = round(self.homogenous_transmission, rounding)
			self.internal_transmission = round(self.internal_transmission, rounding)
			for cat in self.beta_by_category:
				self.beta_by_category[cat] = round(self.beta_by_category[cat], rounding)
			for month in self.monthly_factor:
				self.monthly_factor[month] = round(self.monthly_factor[month], rounding)
	def __repr__(self):
		return "Internal: {} Homogenous: {} Activity Based: {} Monthly Factors: {}".format(self.internal_transmission, self.homogenous_transmission, self.beta_by_category, self.monthly_factor)

	def AsDict(self):
		return {"internal": self.internal_transmission, "homogenous": self.homogenous_transmission, "monthly_factors": self.monthly_factor, "beta_by_category": self.beta_by_category}

def TransmissionParamsFromDict(param_dict):
	return TransmissionParameters(param_dict["beta_by_category"], param_dict["internal"], param_dict["homogenous"], param_dict["monthly_factors"])



def AssessParameters(starting_blocks, disease_stats, start_date, end_date, transmission_parameters, stats_container_by_month, evaluation_intervals, include_interval_blocks):
	transmission_stats_by_month = {
		month: stats_container.GetWeightedStats(
			transmission_parameters.beta_by_category, 
			transmission_parameters.internal_transmission, 
			transmission_parameters.homogenous_transmission,
			transmission_parameters.monthly_factor.get(month, 1.0)) for month, stats_container in stats_container_by_month.iteritems()
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

	total_weighted_error = 0
	total_weight = 0.0
	for i, error in enumerate(prediction_errors):
		total_weighted_error += error * (i + 1)
		total_weight += (i + 1)
	return blocks_by_interval, (total_weighted_error / total_weight)

class UniformRandomParams():
	def __init__(self, internal_range, homogenous_range, cat_range, monthly_range, months):
		self.internal_range = internal_range
		self.homogenous_range = homogenous_range
		self.cat_range = cat_range
		self.monthly_range = monthly_range
		self.months = months

	def GetNewRandom(self):
		internal_transmission = random.uniform(*self.internal_range)
		homogenous_transmission = random.uniform(*self.homogenous_range)
		used_random = random.uniform(*self.cat_range)
		cat_map = {cat: used_random for cat in CATEGORIES}
		monthly_factor = {month: random.uniform(*self.monthly_range) for month in self.months}
		return TransmissionParameters(cat_map, internal_transmission, homogenous_transmission, monthly_factor,  rounding=5)


def DefaultRandomParams(months):
	return UniformRandomParams((0, 1.5), (0, 0), (0, .003), (1.0, 1.0), months)


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

def ComputeRandomParameters(population_by_county, disease_stats, starting_date, ending_date, interconnect_by_month, num_attempts=1000, num_to_keep=10, random_param_gen=None, evaluation_intervals=None):
	def ExtractFeatures(transmission_parameters):
		features = [transmission_parameters.internal_transmission, transmission_parameters.homogenous_transmission, transmission_parameters.beta_by_category['OTHER']]
		# for category, beta in sorted(transmission_parameters.beta_by_category.iteritems()):
		# 	features.append(beta)
		# for month, factor in sorted(transmission_parameters.monthly_factor.iteritems()):
		# 	features.append(factor)
		return features

	print("Generating Training Data")
	features = []
	scores = []

	if random_param_gen is None:
		random_param_gen  = DefaultRandomParams(interconnect_by_month.keys())

	compute_attempt = BuildAssessmentFunction(population_by_county, disease_stats, starting_date, ending_date, interconnect_by_month, evaluation_intervals, include_interval_blocks=False)

	# for i in range(num_attempts):
	# 	random_params = random_param_gen.GetNewRandom()
	# 	_, error = compute_attempt(random_params)
	# 	features.append(ExtractFeatures(random_params))
	# 	scores.append(error)
	# 	if (i % 10) == 0:
	# 		print("Random Attempt: #{} Error: {} Best Error:{}".format(i, error, min(scores)))

	# test_sample_length = int(len(features) * .1)

	# print("Fitting a model with {} training points and {} test points".format(len(features) - test_sample_length, test_sample_length))
	# clf = make_pipeline(StandardScaler(), sklearn.svm.SVR(epsilon=.01))
	# clf.fit(features[test_sample_length:], scores[test_sample_length:])
	# model_fit = clf.score(features[test_sample_length:], scores[test_sample_length:])
	# test_fit = clf.score(features[:test_sample_length], scores[:test_sample_length])
	# print("Model created with training fit {} and test fit {}".format(model_fit, test_fit))

	# num_candidate = 100000
	# print("Generating {} candidate solutions".format(num_candidate))
	# predict_samples = [random_param_gen.GetNewRandom() for i in range(num_candidate)]
	# predict_features = [ExtractFeatures(sample) for sample in predict_samples]
	# predict_scores = clf.predict(predict_features)
	# sorted_scores = sorted((abs(score), i) for i, score in enumerate(predict_scores))

	actual_scores = []
	for i in range(num_attempts):
		# used_params = predict_samples[index]
		used_params = random_param_gen.GetNewRandom()
		_, error = compute_attempt(used_params)
		actual_scores.append((error, used_params))
		if i % 10 == 0:
			print ("Fitted Attempt #{} Predicted Error:{} Error:{} Best Error: {}".format(i, 1.0, error, min(actual_scores)[0]))
	return sorted(actual_scores)[0:num_to_keep]





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


	









