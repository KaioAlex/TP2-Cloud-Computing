from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import pandas as pd

from itertools import groupby
from operator import itemgetter

import csv
import pickle


reader = csv.DictReader(open("./datasets/2023_spotify_ds1.csv"), delimiter=",")
songs_list = []

pids_list = []
for row in reader:
	songs_list.append(row)
	pids_list.append(row['pid'])

pids_list = sorted(set(pids_list), key=int)

data = []
# Display songs grouped by pids
for key, value in groupby(songs_list, key = itemgetter('pid')):
	l = []

	for k in value:
		l.append(k['track_name'])

	data.append(l)

'''
for i in range(0, 10):
	print(i)
	print(data[i])
	print()
exit(-1)
'''

# Transaction encoding
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)

df = pd.DataFrame(te_ary, columns=te.columns_)

# FP-Growth
frequent_itemsets = fpgrowth(df, min_support=0.03, use_colnames=True)

# Association Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.65)
#print(rules)
#print('\n\n')

# User input
#user_input = ['Ride Wit Me', 'Fight Night', 'I Got the Keys'] # songs from different playlists
#user_input = ['Ride Wit Me', 'Red Solo Cup', 'Sweet Emotion'] # songs from the same playlist
user_input = ['Ride Wit Me', 'Fight Night', 'John Cougar, John Deere, John 3:16']
#user_input = ['I Got the Keys']

playlists = []
for i in user_input:
	# Filter rules for relevant recommendations
	user_rules = rules[rules['antecedents'].apply(lambda x: set([i]).issubset(set(x)))]

	# Convert frozensets to sets and extract recommended songs
	recommended_songs = set().union(*(map(set, user_rules['consequents'])))

	# Map row indices to playlists
	playlist_mapping = {i: playlist for i, playlist in enumerate(data)}

	# Correlate recommended songs with playlists
	correlated_playlists = {i: playlist_mapping[i] for i in range(len(data)) if set(recommended_songs).intersection(set(data[i]))}

	playlists.append(correlated_playlists)

#print("Correlated Playlists: ", correlated_playlists.keys())
print(f"Playlists lines: {len(playlists)}")
print(f"Playlists first: {len(playlists[0])}")

# Translate pids of dataset to real pids in .csv
playlists_correct = []
for play in playlists:
	for key in play.keys():
		playlists_correct.append(pids_list[key])

#print(f"Correlated Playlists: {playlists_correct}")
# print(f"Correlated Playlists length: {len(playlists_correct)}")

print(f"Correlated Playlists length uniques: {len(set(playlists_correct))}")
print(f"Correlated Playlists unique sorted: {sorted(set(playlists_correct), key=int)}")

#print(len(playlist_mapping.keys()))