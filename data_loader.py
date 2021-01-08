import datetime
import csv
import gzip
import io


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


def GetDistancingPathForDate(date):
    return "raw_data/placegraph_distancing_data/{year}/{month:02}/{day:02}/{year}-{month:02}-{day:02}-social-distancing.csv.gz".format(
            year=date.year, month=date.month, day=date.day)

def GetDistancingPathsForDateRange(start_date, end_date):
    curr_date = start_date
    paths = []
    while curr_date < end_date:
        paths.append(GetDistancingPathForDate(curr_date))
        curr_date += datetime.timedelta(days=1)
    return paths

def LoadGzippedCsvs(file_names):
    for file_name in file_names:
        print("Loading file {}".format(file_name))
        with io.BufferedReader(gzip.open(file_name)) as f:
            csv_dict_reader = csv.DictReader(f)
            for line in csv_dict_reader:
                yield line


def GetDateRangeForMonth(month_name):
    month_index = MONTH_TO_INDEX[month_name]
    start_date = datetime.datetime(2020, month_index, 1)
    if start_date.month == 12:
        end_date = datetime.datetime(2021, month_index, 1)
    else:
        end_date = datetime.datetime(2020, month_index + 1, 1)
    return start_date, end_date
