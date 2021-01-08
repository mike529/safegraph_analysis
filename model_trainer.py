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
    parser.add_argument("start_month", help="Required: The month to start the training data")
    parser.add_argument("end_month", help="Required: The month to end the training data (inclusive)")
    parser.add_argument("state", help="Required: The state to run the training for")
    parser.add_argument("--num_steps", help="How many steps to run the training grid for", default=30, type=int)
    parser.add_argument("--census_tract_level", help="""If specified compute based on census tract instead of county level (much slower).""", default=False, type=bool)
    parser.add_argument("--monthly_factors", help="""A json dict of month to relative risk factor""", default=None, type=str)
    parser.add_argument("--relative_risks", help="""A json dict of category to relative risk factor.""", default=None, type=str)

    parser.add_argument("--min_internal", help="""Min value for internal transmission.""", default=0.0, type=float)
    parser.add_argument("--max_internal", help="""Max value for internal transmission.""", default=1.5, type=float)

    parser.add_argument("--min_transmission", help="""Min value for activity transmission.""", default=0.0, type=float)
    parser.add_argument("--max_transmission", help="""Max value for activity transmission.""", default=1.0, type=float)

    args = parser.parse_args()

    start_month = MONTH_TO_INDEX[args.start_month]
    end_month = MONTH_TO_INDEX[args.end_month]

    print("Loading monthly interconnect")
    states = args.state.split(',')

    marshal_files_by_month = {}
    for i in range(start_month, end_month+1):
        if args.census_tract_level:
            marshal_files_by_month[i] = ["data/computed_interconnect/{}/{}.marshal".format(MONTH_TO_NAME[i], state) for state in states]
        else:
            marshal_files_by_month[i] = ["data/computed_county_interconnect/{}/{}.marshal".format(MONTH_TO_NAME[i], state) for state in states]

    interconnect_by_month = data_parser.LoadMonthlyMarshalInterconnect(marshal_files_by_month)


    if args.monthly_factors:
        raw_month_factors = json.loads(args.monthly_factors)
        monthly_factors = {MONTH_TO_INDEX[k]: v for k, v in raw_month_factors.iteritems()}
    else:
        monthly_factors = {}

    if args.relative_risks:
        relative_risks = json.loads(args.relative_risks)
    else:
        relative_risks = {}



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

    print("Training for {}-{} and state {}".format(start_date.strftime("%Y/%m/%d"), end_date.strftime("%Y/%m/%d"), args.state))
    best_results = model_evolution.ComputeRandomParameters(pop_by_county, disease_stats, start_date, end_date, interconnect_by_month, args.num_steps,
            evaluation_intervals=evaluation_intervals, relative_risks=relative_risks, monthly_factors=monthly_factors, min_internal=args.min_internal,
            max_internal=args.max_internal, min_transmission=args.min_transmission, max_transmission=args.max_transmission)

    for i, (error, transmission_params) in enumerate(best_results):
        formatted = json.dumps(transmission_params.AsDict())
        print("Option #{}: {} \nError:{}\nFormatted:{}\n".format(i+1, transmission_params, error, formatted))
