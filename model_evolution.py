from __future__ import division
import numpy as np
import collections
import csv
import data_parser
import datetime
import copy


INFECTION_DURATION = 14

CATEGORIES = ['EDUCATION', 'HEALTH_CARE', 'OTHER', 'RELIGION', 'RESTAURANT']

RELATIVE_POP_WEIGHTS = [
        (.20, .8),
        (.8, .2),
]

def NormalizeWeights(relative_weights):
    return relative_weights

class ParallelBlockStats():
    def __init__(self, interconnect_data, disease_by_county, population_by_county):
        total_normalization_by_county = collections.defaultdict(float)

        for interconnect in interconnect_data:
            total_normalization_by_county[GetCounty(interconnect.block)] += interconnect.normalization_factor

        normalized_weights = NormalizeWeights(RELATIVE_POP_WEIGHTS)
        matrix_size = len(interconnect_data) * len(normalized_weights)
        self.block_counties = []
        self.populations = np.zeros(matrix_size)
        self.current_index = 0
        self.current_sick_by_day = np.zeros((INFECTION_DURATION, matrix_size))
        self.current_sick = np.zeros(matrix_size)
        self.susceptible = np.zeros(matrix_size)
        current_index = 0
        for interconnect in interconnect_data:
            county = GetCounty(interconnect.block)
            for (relative_pop, relative_transmission) in normalized_weights:
                self.block_counties.append(county)
                ever_sick, current_sick = disease_by_county.get(county, (0, None))
                county_population = population_by_county.get(county, 1)

                total_normalization_factor = total_normalization_by_county[county]

                fractional_part = (interconnect.normalization_factor / total_normalization_factor) * relative_pop
                population = max(county_population * fractional_part, 5)
                self.populations[current_index] = population
                self.susceptible[current_index] = min(population, max((county_population - ever_sick) * fractional_part, 0))

                if current_sick:
                    for day, sick in enumerate(current_sick):
                        self.current_sick_by_day[day][current_index] = sick * fractional_part
                        self.current_sick[current_index] += sick * fractional_part
                current_index += 1

        # Helper variables
        # Probablity of a random infection missing a particular person.
        self.non_infection_probs = 1 - 1 / self.populations
        self.population_fractions = self.populations / sum(self.populations)


    def ComputeNewInfections(self, raw_infections):
        # Prob of not being infected by any of the infections in the group.
        prob_not_infected = np.power(self.non_infection_probs, raw_infections / INFECTION_DURATION)
        # Only those susceptible can be infected.
        return (1 - prob_not_infected) * self.susceptible


    def AdvanceDay(self, new_infections):
        recovered = self.current_sick_by_day[self.current_index]
        self.current_sick = (self.current_sick - recovered + new_infections)
        self.current_sick_by_day[self.current_index] = new_infections
        # Update to the next index this avoids having to do an array copy.
        self.current_index = (self.current_index + 1) % INFECTION_DURATION
        # Once infected you are immune.
        self.susceptible -= new_infections


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
    def __init__(self, transmission_per_block, transmission_matrix, internal_matrix, internal_transmission=0, homogenous_transmission=0, transmission_multiplier=1.0):
        self.transmission_per_block = transmission_per_block
        self.transmission_matrix = transmission_matrix
        self.internal_matrix = internal_matrix
        self.internal_transmission = internal_transmission
        self.homogenous_transmission = homogenous_transmission
        self.transmission_multiplier = transmission_multiplier

class FastStatsContainer():
    def __init__(self, raw_interconnect_data, relative_weights=None):
        if relative_weights is None:
            relative_weights = {}

        normalized_weights = NormalizeWeights(RELATIVE_POP_WEIGHTS)
        matrix_size = len(raw_interconnect_data) * len(normalized_weights)

        self.other_weight = relative_weights.get('OTHER', 1.0)
        self.block_length = len(raw_interconnect_data)
        self.transmission_per_block = np.zeros(matrix_size)
        self.transmission_matrix = np.zeros((matrix_size, matrix_size))
        self.internal_matrix = np.zeros((matrix_size, matrix_size))



        def UpdateTransmissionPerBlock(block_index, update_base):
            for i, (relative_pop, relative_transmission) in enumerate(normalized_weights):
                current_index = block_index * len(normalized_weights) + i
                self.transmission_per_block[current_index] += (relative_transmission / relative_pop) * update_base

        def UpdateInternalTransmission(block_index):
            for i, (relative_pop, _) in enumerate(normalized_weights):
                start_index = block_index * len(normalized_weights) + i
                for j, (relative_dest_pop, _) in enumerate(normalized_weights):
                    dest_index = block_index * len(normalized_weights) + j
                    self.internal_matrix[start_index][dest_index] = relative_dest_pop

        def UpdateNeighborTransmission(block_index, neighbor_index, update_base):
            for i, (_, relative_transmission) in enumerate(normalized_weights):
                start_index = block_index * len(normalized_weights) + i
                for j, (relative_dest_pop, relative_dest_transmission) in enumerate(normalized_weights):
                    dest_index = neighbor_index * len(normalized_weights) + j
                    self.transmission_matrix[start_index][dest_index] += relative_dest_transmission * update_base

        index_for_block = {}

        for i,raw_data in enumerate(raw_interconnect_data):
            index_for_block[raw_data.block] = i


        for i, interconnect in enumerate(raw_interconnect_data):
            UpdateInternalTransmission(i)
            for category, raw_category_data in interconnect.category_map.iteritems():
                UpdateTransmissionPerBlock(i, (raw_category_data.transmission * relative_weights.get(category, 1.0)))
                # self.transmission_per_block[i] = 1
                for neighbor, transmission_fraction in raw_category_data.neighbors.iteritems():
                    if neighbor in index_for_block:
                        UpdateNeighborTransmission(i, index_for_block[neighbor], (transmission_fraction * relative_weights.get(category, 1.0)))
        # self.transmission_matrix = np.matmul(self.transmission_matrix, self.transmission_matrix.transpose())
        for i in range(len(self.transmission_matrix)):
            row_sum = np.sum(self.transmission_matrix[i])
            if row_sum != 0:
                self.transmission_matrix[i] /= row_sum

    def GetWeightedStats(self, beta_by_category, internal_transmission=0, homogenous_transmission=0, monthly_factor=1.0):
        transmission_multiplier = beta_by_category.get('OTHER', 1.0) * monthly_factor / self.other_weight
        return TransmissionStatistics(self.transmission_per_block, self.transmission_matrix, self.internal_matrix,
                internal_transmission=internal_transmission * monthly_factor,
                homogenous_transmission=homogenous_transmission * monthly_factor,
                transmission_multiplier=transmission_multiplier)



def AdvanceDayParallel(parallel_block_stats, transmission_stats):
    homogenous_transmission = parallel_block_stats.population_fractions * sum(parallel_block_stats.current_sick) * transmission_stats.homogenous_transmission
    internal_transmission =  np.matmul(parallel_block_stats.current_sick * transmission_stats.internal_transmission, transmission_stats.internal_matrix)
    # internal_transmission = parallel_block_stats.current_sick * transmission_stats.internal_transmission
    transmitted_infections = parallel_block_stats.current_sick * transmission_stats.transmission_per_block * transmission_stats.transmission_multiplier
    received_infections = np.matmul(transmitted_infections, transmission_stats.transmission_matrix)

    actual_new_infections = parallel_block_stats.ComputeNewInfections(internal_transmission + homogenous_transmission + received_infections)
    # import pdb; pdb.set_trace()
    parallel_block_stats.AdvanceDay(actual_new_infections)

def GetCounty(block_id):
    return block_id[:5]

# Computes county level stats for a set of blocks.
def GetStatsByCounty(blocks):
    pop_by_county = collections.defaultdict(float)
    predicted_ever_sick_by_county = collections.defaultdict(float)
    current_sick_by_county = collections.defaultdict(float)
    for i in range(len(blocks.block_counties)):
        county = blocks.block_counties[i]
        pop_by_county[county] += blocks.populations[i]
        predicted_ever_sick_by_county[county] += (blocks.populations[i] - blocks.susceptible[i])
        current_sick_by_county[county] += blocks.current_sick[i]

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
        actual_ever, actual_current = data_parser.ExtractStats(disease_stats.get(county, None), current_date, INFECTION_DURATION)
        relative_error = GetRelativeError(predicted_ever_sick, actual_ever)
        pop_weight = pop
        total_weight += pop_weight
        total_weighted_error += pop_weight * relative_error
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

    copied_blocks = copy.deepcopy(starting_blocks)

    prediction_errors = []
    blocks_by_interval = {}
    if include_interval_blocks:
        blocks_by_interval[current_date] = copy.deepcopy(copied_blocks)
    while current_date < end_date:
        AdvanceDayParallel(copied_blocks, transmission_stats_by_month[current_date.month])
        current_date += datetime.timedelta(days=1)
        if current_date in evaluation_intervals or current_date == end_date:
            prediction_errors.append(ComputeWeightedError(copied_blocks, disease_stats, current_date))
            if include_interval_blocks:
                blocks_by_interval[current_date] = copy.deepcopy(copied_blocks)

    total_weighted_error = 0
    total_weight = 0.0
    for i, error in enumerate(prediction_errors):
        total_weighted_error += error
        total_weight += 1
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
    return UniformRandomParams((0, 1.5), (0, 0), (0, .05), (1.0, 1.0), months)


def BuildAssessmentFunction(population_by_county, disease_stats, starting_date, ending_date, interconnect_by_month, stats_container_by_month, evaluation_intervals=None, include_interval_blocks=False):
    if evaluation_intervals is None:
        evaluation_intervals = set(datetime.datetime(2020, month, 1) for month in range(starting_date.month + 1, ending_date.month + 1))


    starting_disease_by_county = data_parser.LoadDiseaseStatsByCounty(disease_stats, starting_date, INFECTION_DURATION)

    starting_blocks = ParallelBlockStats(interconnect_by_month[starting_date.month], starting_disease_by_county, population_by_county)


    def ComputeAttempt(transmission_parameters):
        return AssessParameters(starting_blocks, disease_stats, starting_date, ending_date, transmission_parameters, stats_container_by_month, evaluation_intervals, include_interval_blocks)
    return ComputeAttempt

def ComputeRandomParameters(population_by_county, disease_stats, starting_date, ending_date, interconnect_by_month, num_steps=30,
        num_to_keep=10, evaluation_intervals=None, relative_risks=None, monthly_factors=None,
        min_internal=0,
        max_internal=1.5,
        min_transmission=0,
        max_transmission=1.0):
    if relative_risks is None:
        relative_risks = {}
    if monthly_factors is None:
        monthly_factors = {}

    print("Generating Training Data")

    stats_container_by_month = {month:FastStatsContainer(interconnect, relative_weights=relative_risks) for month, interconnect in interconnect_by_month.iteritems()}

    compute_attempt = BuildAssessmentFunction(population_by_county, disease_stats, starting_date, ending_date, interconnect_by_month, stats_container_by_month, evaluation_intervals, include_interval_blocks=False)

    def GetTransmissionParams(internal, cat_weight):
        cat_map = {cat: cat_weight * relative_risks.get(cat, 1.0) for cat in CATEGORIES}
        return TransmissionParameters(cat_map, internal, 0, monthly_factors, rounding=None)


    actual_scores = []

    i = 0
    for internal in np.linspace(min_internal, max_internal, num=num_steps):
        for cat in np.linspace(min_transmission, max_transmission, num=num_steps):

            used_params = GetTransmissionParams(internal, cat)
            _, error = compute_attempt(used_params)
            actual_scores.append((error, used_params))
            if i % 10 == 0:
                print ("Attempt: {} Internal {}, Transmission {}, Error:{} Best Error: {}".format(i, internal, cat, error, min(actual_scores)[0]))
            i+=1
    return sorted(actual_scores)[0:num_to_keep]



def RunScenario(population_by_county, disease_stats, starting_date, ending_date, interconnect_by_month, transmission_parameters, evaluation_intervals=None):
    stats_container_by_month = {month:FastStatsContainer(interconnect) for month, interconnect in interconnect_by_month.iteritems()}


    compute_attempt = BuildAssessmentFunction(population_by_county, disease_stats, starting_date, ending_date, interconnect_by_month, stats_container_by_month, evaluation_intervals, include_interval_blocks=True)

    computed_blocks, error = compute_attempt(transmission_parameters)

    predicted_final_blocks = computed_blocks[ending_date]
    predicted_stats_by_county = GetStatsByCounty(predicted_final_blocks)

    starting_disease_by_county = data_parser.LoadDiseaseStatsByCounty(disease_stats, starting_date, INFECTION_DURATION)
    final_disease_by_county = data_parser.LoadDiseaseStatsByCounty(disease_stats, ending_date, INFECTION_DURATION)

    return computed_blocks, error
