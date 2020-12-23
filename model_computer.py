import argparse
import model_evolution
import data_parser
import datetime

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
	parser.add_argument("transmission_params", help="""Required: A comma separated string with float values for the parameters. The format should be
		internal_transmission, homogenous_transmission,EDUCATION_beta, HEALTH_CARE_beta, OTHER_beta, RELIGION_beta, RESTAURANT_beta""")

	args = parser.parse_args()

	start_month = MONTH_TO_INDEX[args.start_month]
	end_month = MONTH_TO_INDEX[args.end_month]

	raw_params = [float(x) for x in args.transmission_params.split(",")]
	internal_transmission = raw_params[0]
	homogenous_transmission = raw_params[1]
	cat_map = {}
	for i, cat in enumerate(model_evolution.CATEGORIES):
		cat_map[cat] = raw_params[2 + i]
	transmission_params = model_evolution.TransmissionParameters(cat_map, internal_transmission, homogenous_transmission)

	print("Loading monthly interconnect")
	interconnect_by_month = {}
	for i in range(start_month, end_month+1):
		interconnect_by_month[i] = data_parser.LoadPickle("data/computed_interconnect/{}/{}.pickle".format(MONTH_TO_NAME[i], args.state)).values()


	print("Loading population and disease estimate data")
	pop_by_county = data_parser.LoadPopulationByCounty("data/county_pops.pickle")
	disease_stats = data_parser.LoadRawDiseaseStats("data/county_estimates.pickle")

	start_date = datetime.datetime(2020, start_month, 1)
	if end_month == 12:
		end_date = datetime.datetime(2021, 1, 1)
	else:
		end_date = datetime.datetime(2020, end_month + 1, 1)

	print("Running scenario for {}-{}, State={}, Transmission Parameters={}".format(start_date.strftime("%Y/%M/%d"), end_date.strftime("%Y/%M/%d"), args.state, transmission_params))
	computed_blocks, results, error = model_evolution.RunScenario(pop_by_county, disease_stats, start_date, end_date, interconnect_by_month, transmission_params)

	print("Overall Error: {}".format(error))
	predicted_total = sum(result['ever_sick']['predicted'] for result in results.values())
	actual_total = sum(result['ever_sick']['finish'] for result in results.values())
	print("Predicted Total Infections: {} Actual: {}".format(predicted_total, actual_total))

