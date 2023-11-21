from fpgrowth_py import fpgrowth
import csv
import pickle

from itertools import groupby
from operator import itemgetter

reader = csv.DictReader(open("./datasets/2023_spotify_ds1.csv"), delimiter=",")

songs_list = []

for row in reader:
	songs_list.append(row)

itemSetList = []

# Display songs grouped by pids
for key, value in groupby(songs_list, key = itemgetter('pid')):
	l = []
	#print(key)

	for k in value:
		l.append(k['track_name'])
		#print(k)

	itemSetList.append(l)

'''
itemSetList = [['eggs', 'bacon', 'soup'],
               ['eggs', 'bacon', 'apple'],
               ['soup', 'bacon', 'banana']]
'''

freqItemSet, rules = fpgrowth(itemSetList, minSupRatio=0.07, minConf=0.05)
print(freqItemSet)
print(rules)
