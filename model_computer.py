import argparse
import model_evolution
import data_parser
import datetime
import json

MONTH_TO_INDEX = {
	"january": 1,
	"jan": 1,
	"february": 2,
	"feb": 2,
	"march": 3,
	"mar": 3,
	"april": 4,
	"apr": 4,
	"may": 5,
	"june": 6,
	"jun": 6,
	"july": 7,
	"jul": 7,
	"august": 8,
	"aug": 8,
	"september": 9,
	'sep': 9,
	"sept": 9,
	"october": 10,
	"oct": 10,
	"november": 11,
	"nov": 11,
	"december": 12,
	"dec": 12,
}

MONTH_TO_NAME = [None, "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("start_month", help="Required: The month to start the simulation")
	parser.add_argument("end_month", help="Required: The month to end the simulation (inclusive)")
	parser.add_argument("state", help="Required: The state to run the simulation for")
	parser.add_argument("transmission_params", help="""Required: A json formatted specification of the transmission params. Must match what is generated by TransmissionParameters.AsDict()""")
	parser.add_argument("--census_tract_level", help="""If specified compute based on census tract instead of county level (slower).""",  default=False, type=bool)

	args = parser.parse_args()

	start_month = MONTH_TO_INDEX[args.start_month]
	end_month = MONTH_TO_INDEX[args.end_month]

	transmission_params = model_evolution.TransmissionParamsFromDict(json.loads(args.transmission_params))

	print("Loading monthly interconnect")
	marshal_file_by_month = {}
	for i in range(start_month, end_month+1):
		if args.census_tract_level:
			marshal_file_by_month[i] = "data/computed_interconnect/{}/{}.marshal".format(MONTH_TO_NAME[i], args.state)
		else:
			marshal_file_by_month[i] = "data/computed_county_interconnect/{}/{}.marshal".format(MONTH_TO_NAME[i], args.state)

	interconnect_by_month = data_parser.LoadMonthlyMarshalInterconnect(marshal_file_by_month)


	print("Loading population and disease estimate data")
	pop_by_county = data_parser.LoadPopulationByCounty("data/county_pops.pickle")
	disease_stats = data_parser.LoadRawDiseaseStats("data/county_estimates.pickle")

	start_date = datetime.datetime(2020, start_month, 1)
	if end_month == 12:
		end_date = datetime.datetime(2021, 1, 1)
	else:
		end_date = datetime.datetime(2020, end_month + 1, 1)

	evaluation_intervals = set()
	curr_date = start_date
	while curr_date < end_date:
		evaluation_intervals.add(curr_date)
		curr_date += datetime.timedelta(days=7)
	evaluation_intervals.add(end_date)
	print("Running scenario for {}-{}, State={}, Transmission Parameters={}".format(start_date.strftime("%Y/%m/%d"), end_date.strftime("%Y/%m/%d"), args.state, transmission_params))
	computed_blocks, results, error = model_evolution.RunScenario(pop_by_county, disease_stats, start_date, end_date, interconnect_by_month, transmission_params, evaluation_intervals)

	print("Overall Error: {}".format(error))

	header = ["Date", "Predicted Ever Infected", "Predicted Current Infected", "Observed Ever Infected", "Observed Current Infected", "Error"]
	rows = []

	for date, blocks_for_date in sorted(computed_blocks.items()):
		counties = set(model_evolution.GetCounty(block.block_id) for block in blocks_for_date)
		sick = 0
		current_sick = 0
		for county in counties:
			county_sick, county_infections = data_parser.ExtractStats(disease_stats[county], date, model_evolution.INFECTION_DURATION)
			sick += county_sick
			current_sick += sum(county_infections)

		rows.append(
			[
				date.strftime("%Y/%m/%d"), 
				round(sum(block.ever_infected for block in blocks_for_date), 2),
				round(sum(block.current_infected for block in blocks_for_date), 2),
				round(sick, 2), 
				round(current_sick, 2),
				round(sum(block.ever_infected for block in blocks_for_date) / sick, 2)
			]
		)

	print(",".join(header))
	print("\n".join([",".join([str(x) for x in row]) for row in rows]))

