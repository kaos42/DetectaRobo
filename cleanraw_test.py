## clean raw data

import datetime
import pandas
from collections import defaultdict
from numpy import mean, nan

datadir = 'data/'
infile = datadir + 'raw/ftc_testing_set_Corrected.csv'
outfile = datadir + 'clean/test.csv'

# read training file, format as a dictionary
with open(infile, 'r') as f:
    inlines = [l.strip().split(',') for l in f.readlines()]
header = inlines[0]
header[3] = 'DateTime' # rename 'Date-Time' to 'DateTime'
data = [dict(zip(header, l)) for l in inlines[1:]]

# setup of time-of-day comparisons
q1 = range(0, 15)
q2 = range(15, 30)
q3 = range(30, 45)
night = range(0, 7)
morning = range(7, 12)
afternoon = range(12, 17)
evening = range(17, 24)

# format area-code lookup
with open(datadir + 'area_state', 'r') as f, open(datadir + 'area_territories', 'r') as g, open(datadir + 'area_special', 'r') as h:
    areas = dict([l.strip().split("\t\t") for l in f.readlines()][1:] + [[l.strip(), 'Territory'] for l in g.readlines()][1:] + [[l.strip(), 'Special'] for l in h.readlines()][1:])

# data structures for collecting data
callcounts = defaultdict(lambda: {'in': 0, 'out': 0, 'ratio': 0})
yesrobo = set([])
norobo = set([])
call_by_number = defaultdict(list)

# first-pass data cleaning
for d in data:
    d.pop('Ref. No.') # remove index
    # d['Robocall'] = int(d['Robocall'])

    # record both calls for each number
    callcounts[d['To']]['in'] = callcounts[d['To']]['in'] + 1
    callcounts[d['From']]['out'] = callcounts[d['From']]['out'] + 1

    # # record if 'From' is a robocall
    # if d['Robocall'] == 1:
    #     yesrobo.add(d['From'])
    # else:
    #     norobo.add(d['From'])

    # phone number features
    for field in ['From', 'To']:
        # all country codes are 1
        # d[field + 'Country'] = int(d[field][:1])
        d[field + 'AreaCode'] = int(d[field][1:4])
        d[field + 'Exchange'] = int(d[field][4:7])
        try:
            d[field + 'State'] = areas[str(d[field + 'AreaCode'])]
        except KeyError:
            d[field + 'State'] = 'Missing'

    if d['FromAreaCode'] == d['ToAreaCode']:
        d['SameArea'] = 1
    else:
        d['SameArea'] = 0

    if d['FromAreaCode']+d['FromExchange'] == d['ToAreaCode']+d['ToExchange']:
        d['SameAreaExchange'] = 1
    else:
        d['SameAreaExchange'] = 0

    if d['FromState'] == d['ToState']:
        d['SameState'] = 1
    else:
        d['SameState'] = 0

    # all calls are from valid phone numbers
    # if str(d['FromAreaCode'])[:1] in ['0', '1'] or str(d['FromExchange'])[:1] in ['0', '1']:
    #     d['FromInvalid'] = 1
    # else:
    #     d['FromInvalid'] = 0

    # date features
    timestamp = datetime.datetime.strptime(d['DateTime'], '%Y-%m-%d %H:%M:%S')
    d['DoW'] = timestamp.weekday() # 0 == monday
    call_by_number[d['From']].append(timestamp)

    # Year and Month are not useful: in training set, all have year 2014, month 11 or 12
    # d['Year'] = timestamp.year
    # d['Month'] = timestamp.month
    d['Day'] = timestamp.day
    d['Hour'] = timestamp.hour
    d['Minute'] = timestamp.minute
    d['Second'] = timestamp.second

    # which quarter of the hour?
    if d['Minute'] in q1:
        d['Quarter'] = 1
    elif d['Minute'] in q2:
        d['Quarter'] = 2
    elif d['Minute'] in q3:
        d['Quarter'] = 3
    else:
        d['Quarter'] = 4

    # which time of day?
    if d['Hour'] in night:
        d['TimeOfDay'] = 0
    elif d['Hour'] in morning:
        d['TimeOfDay'] = 1
    elif d['Hour'] in afternoon:
        d['TimeOfDay'] = 2
    else:
        d['TimeOfDay'] = 3

# out / in ratio of calls
for number, count in callcounts.iteritems():
    if count['in'] == 0:
        count['ratio'] = -1
    else:
        count['ratio'] = float(count['out']) / count['in']

# calculate difference between calls
call_deltas = defaultdict(list)
for number, times in call_by_number.iteritems():
    for i in xrange(1, len(times)):
        td = times[i] - times[i-1]
        call_deltas[number].append(td.total_seconds())

avg_delta = defaultdict(lambda: 0)
for number, deltas in call_deltas.iteritems():
    avg_delta[number] = mean(deltas)

# second pass over data---update with further info
for d in data:
    d['FromToRatio'] = callcounts[d['From']]['ratio']
    d['AvgDelta'] = avg_delta[d['From']]
    d['FromCallVolume'] = callcounts[d['From']]['out'] + callcounts[d['From']]['in']
    # if d['From'] in yesrobo:
    #     d['RoboEver'] = 1
    # else:
    #     d['RoboEver'] = 0

# get rid of unwanted columns
for d in data:
    d.pop("DateTime")
    d.pop("From")
    d.pop("To")
    d.pop("FromToRatio")
    d.pop("FromExchange")
    d.pop("ToExchange")
    d.pop("FromAreaCode")
    d.pop("ToAreaCode")

# output to csv for analysis in R
pandas.DataFrame.from_dict(data).to_csv(outfile, index = False)
