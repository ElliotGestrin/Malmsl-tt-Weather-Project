# Merges the data from the various readings into a single CSV file
# Two options, either before "Nederbörd" started to be measured or after
# Change with if True/False below

from csv import DictReader, DictWriter
from datetime import datetime, timedelta
from more_itertools import peekable
from tqdm.auto import tqdm

def time(string):
    return datetime.strptime(string,"%Y-%m-%d %H:%M:%S")

def read_time(reading):
    return time(reading['Datum']+" "+reading['Tid (UTC)'])

def peek_time(reader):
    return read_time(reader.peek())

for set in ["early","late","full"]:
    if set == "early":
        start = time("1951-01-01 01:00:00")
        end = time("2013-01-01 00:00:00")
        files = [
            ("Lufttemperatur.csv", ["Lufttemperatur"]),
            ("Lufttryck.csv", ["Lufttryck reducerat havsytans nivå"]),
            ("RelativLuftfuktighet.csv", ["Relativ Luftfuktighet"]),
            ("Sikt.csv", ["Sikt"]), 
            ("TotalMolnmängd.csv", ["Total molnmängd"]),
            ("Vind.csv", ["Vindriktning", "Vindhastighet"])
        ]
        output = "MalmslättEarly.csv"
    elif set == "late":
        start = time("2013-01-01 01:00:00")
        end = time("2022-11-01 00:00:00")
        files = [
            ("Lufttemperatur.csv",["Lufttemperatur"]),
            ("Lufttryck.csv",["Lufttryck reducerat havsytans nivå"]),
            ("RelativLuftfuktighet.csv",["Relativ Luftfuktighet"]),
            ("Sikt.csv",["Sikt"]),
            ("TotalMolnmängd.csv",["Total molnmängd"]),
            ("Vind.csv",["Vindriktning","Vindhastighet"]),
            ("Nederbörd.csv",["Nederbördsmängd"])
        ]
        output = "MalmslättLate.csv"
    elif set == "full":
        start = time("1951-01-01 01:00:00")
        end = time("2022-11-01 00:00:00")
        files = [
            ("Lufttemperatur.csv", ["Lufttemperatur"]),
            ("Lufttryck.csv", ["Lufttryck reducerat havsytans nivå"]),
            ("RelativLuftfuktighet.csv", ["Relativ Luftfuktighet"]),
            ("Sikt.csv", ["Sikt"]), 
            ("TotalMolnmängd.csv", ["Total molnmängd"]),
            ("Vind.csv", ["Vindriktning", "Vindhastighet"])
        ]
        output = "MalmslättFull.csv"

    open_files = [open(f,'r',encoding="utf-8-sig") for f,t in files]
    open_output = open(output,'w', newline='',encoding="utf-8-sig")
    categories = ["Month","Day","Hour","Year"] + [t for f, ts in files for t in ts]
    readers = [peekable(DictReader(of,delimiter=';')) for of in open_files]
    writer = DictWriter(open_output,categories)
    writer.writeheader()
    targets = [t for f,t in files]
    last_readings = []
    curr = start

    # Find the first value needed
    for reader in readers:
        reading = next(reader)
        while(peek_time(reader) < start):
            reading = next(reader)
        last_readings.append(reading)

    # Build the connected data
    time_span = end - start
    with tqdm(total=int(time_span.days * 24 + time_span.seconds/3600), colour="green") as pbar:
        while(curr < end):
            value = {
                'Month': curr.month,
                'Day': curr.day,
                'Hour': curr.hour if curr.hour != 0 else 24,
                'Year': curr.year
            }

            for i, reader in enumerate(readers):
                # Discard useless data (shouldn't really happen more than once per step)
                while(peek_time(reader) < curr):
                    last_readings[i] = next(reader)

                # Missing data
                if "" in [last_readings[i].get(target) for target in targets[i]]+[reader.peek().get(target) for target in targets[i]]:
                    break
                # Known data
                if peek_time(reader) == curr:
                    value = value | {target : reader.peek()[target] for target in targets[i]}
                # Interpolated data
                else:
                    dT = (curr - read_time(last_readings[i])) / (peek_time(reader) - read_time(last_readings[i]))
                    lin = {
                        target: float(last_readings[i].get(target)) + dT*(float(reader.peek().get(target))-float(last_readings[i].get(target))) for target in targets[i]
                    }
                    value = value | lin
            else: # If no missing data
                writer.writerow(value)
            curr += timedelta(hours=1)
            pbar.update(1)