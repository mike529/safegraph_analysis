from __future__ import division

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
import data_loader
import io

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
        if neighbor not in self.neighbors:
            self.neighbors[neighbor] = 0
        self.neighbors[neighbor] += square_pings
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

    top_level_code = naics_code // 10000
    if top_level_code == 61:
        return "EDUCATION"
    elif top_level_code == 72:
        return "RESTAURANT"
    elif top_level_code == 62:
        return "HEALTH_CARE"
    elif naics_code // 1000 == 813:
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
    parent_ids = set()
    poi_to_category = {}
    for file_name in file_names:
        print("Parsing file: {}".format(file_name))
        with io.BufferedReader(gzip.open(file_name)) as f:
            csv_dict_reader = csv.DictReader(f)
            for i, row in enumerate(csv_dict_reader):
                poi_to_category[row['safegraph_place_id']] = category_extractor(int(row['naics_code'] or 0))
                parent_id = row['parent_safegraph_place_id']
                if parent_id:
                    parent_ids.add(parent_id)
    for parent_id in parent_ids:
        poi_to_category[parent_id] = None
    return poi_to_category

def GetRowAdjustmentFactor(poi_row):

    def GetDensity(array):
        square_sum = sum(x * x for x in array)
        total = sum(x for x in array)
        return len(array) * square_sum / (total * total)

    visits_by_day = json.loads(poi_row['visits_by_day'])
    pop_by_hour = json.loads(poi_row['popularity_by_hour'])

    # Potentially can do something with average based on buckets rather than median.
    dwell_time = (sum(pop_by_hour) / float(poi_row['raw_visit_counts'])) - 1

    dwell_time = max(dwell_time, .1)


    num_days = len(visits_by_day)
    # We have NUM_DEVICES per home_cbg but we want NUM_VISITS

    home_cbgs = json.loads(poi_row['visitor_home_cbgs'])
    total_homes = sum(x for x in home_cbgs.values())
    # if total_homes:
    #       return float(poi_row['raw_visit_counts']) / total_homes
    # else:
    #       return 0

    if total_homes:
        visit_multiplier = float(poi_row['raw_visit_counts']) / total_homes
    else:
        visit_multiplier = 1
    # We have data on visits by hour and by day.
    hourly_density_multiplier = GetDensity(pop_by_hour)
    daily_density_multiplier = GetDensity(visits_by_day)

    return visit_multiplier /  num_days
    # return (dwell_time * visit_multiplier * hourly_density_multiplier * daily_density_multiplier) / num_days

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


    for i, row in enumerate(data_loader.LoadGzippedCsvs(file_names)):
        if i % 100000 == 0:
            print("Loading row #{}: POI:{}".format(i, row['location_name']))
        poi_state = GetState(row['poi_cbg'])
        poi_id = i
        poi_type = poi_to_category.get(row['safegraph_place_id'], None)
        if not poi_type:
            continue
        adjustment_factor = GetRowAdjustmentFactor(row)
        for source_location, source_ping_count in json.loads(row['visitor_home_cbgs']).iteritems():
            # Drop low signal pings.
            if source_ping_count < 5:
                continue
            source_state = GetState(source_location)
            if filter_states and source_state not in filter_states:
                continue
            source_id = GetId(source_location)
            MarkKey(source_state, poi_type, source_id, (source_ping_count * adjustment_factor), pings_by_neighborhood)

            if poi_state == source_state:
                MarkKey(source_state, poi_type, (source_id, poi_id), (source_ping_count * adjustment_factor), visits)
    return pings_by_neighborhood, visits

def BuildPoiSerialization(pings_for_state, visits_for_state, census_block_stats, row_key_extractor):
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
        neighborhood_stats[neighborhood].total_pings = count / population
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
    for i, row in enumerate(data_loader.LoadGzippedCsvs(poi_pattern_file_names)):
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


class CountyDistancingStats():
    def __init__(self):
        self.devices = 0
        self.device_non_home_minutes = 0
        self.neighbors = {}

def GetTravelFromMean(row):
    return float(row['mean_non_home_dwell_time'])


# Empirically computed by comparing 200000 lines where both are populated
FUDGE_FACTOR = .95

def GetTravelFromBuckets(row):
    def ParsePart(part):
        internal_parts = part.split('-')
        if len(internal_parts) == 2:
            return (int(internal_parts[0]) - 1, int(internal_parts[1]))
        else:
            return (0, int(internal_parts[0][1:]))
    total_count = 0
    total_time = 0
    for part, count in json.loads(row['bucketed_away_from_home_time']).iteritems():
        part_range = ParsePart(part)
        total_count += count
        total_time += count * (part_range[0] + ((part_range[1] - part_range[0]) / 2))
    return (total_time / total_count) * FUDGE_FACTOR




def LoadSocialDistancingStats(file_names, census_block_stats, row_key_extractor, filter_states):
    def GetId(area):
        converted = row_key_extractor(area)
        return converted

    def GetState(area):
        # Uncomment to generate a single national model.
        # return 'us'
        census_block_stat = census_block_stats.get(row_key_extractor(area))
        if not census_block_stat:
            return None
        return census_block_stat['state']



    county_counts = collections.defaultdict(lambda: collections.defaultdict(CountyDistancingStats))
    for i, row in enumerate(data_loader.LoadGzippedCsvs(file_names)):
        if i % 100000 == 0:
            print("Loading row #{}: CBG:{}".format(i, row['origin_census_block_group']))
        source_state = GetState(row['origin_census_block_group'])
        if filter_states and source_state not in filter_states:
            continue

        # Before May 18th there is non mean_non_home_dwell_time so we need to adjust.
        if 'mean_non_home_dwell_time' in row:
            travel =  GetTravelFromMean(row)
        else:
            travel = GetTravelFromBuckets(row)
        devices = int(row['device_count'])
        source_id = GetId(row['origin_census_block_group'])
        distancing_stats = county_counts[source_state][source_id]
        distancing_stats.devices += devices
        distancing_stats.device_non_home_minutes += (devices * travel)
        for destination, count in json.loads(row['destination_cbgs']).items():
            dest_state = GetState(destination)
            destination_id = GetId(destination)
            if dest_state == source_state:
                current = distancing_stats.neighbors.get(destination_id, 0)
                distancing_stats.neighbors[destination_id] = current + count
    return county_counts

def ConvertSocialDistancingStats(social_distancing_by_state_block):
    interconnect_map = {}
    for state, social_distancing_by_block in social_distancing_by_state_block.iteritems():
        interconnect_state_map = {}
        for block, social_distancing in social_distancing_by_block.iteritems():
            transmission = data_parser.CategoryData(social_distancing.device_non_home_minutes / (social_distancing.devices * 60), social_distancing.neighbors)
            interconnect_data = data_parser.InterconnectData(block, social_distancing.devices, {'OTHER': transmission})
            interconnect_state_map[block] = interconnect_data
        interconnect_map[state] = interconnect_state_map
    return interconnect_map

def SplitIntoSerializedSocialDistancingStates(social_distancing_file_names, census_block_stats_file_name, row_key_extractor=ConvertRowKey, filter_states=None):
    print("Joining Census Data")
    census_block_stats = GetCensusBlockStats(census_block_stats_file_name, row_key_extractor)
    print("Loading Social Distancing Data")
    social_distancing_data = LoadSocialDistancingStats(social_distancing_file_names, census_block_stats, row_key_extractor, filter_states)
    print("Converting to interconnect data")
    converted_data = ConvertSocialDistancingStats(social_distancing_data)
    return converted_data


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
            state_stats[category] = BuildPoiSerialization(pings.get(state, {}).get(category, {}), visits.get(state, {}).get(category, {}), census_block_stats, row_key_extractor=row_key_extractor)
        print("Serializing")
        serialized = ConvertToInterconnectData(state_stats)
        combined_data_by_state[state] = serialized

    return combined_data_by_state
