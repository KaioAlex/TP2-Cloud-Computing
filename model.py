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

#data = {}
data = []
# Display songs grouped by pids
for key, value in groupby(songs_list, key = itemgetter('pid')):
	l = []

	for k in value:
		l.append(k['track_name'])

	data.append(l)
	#data[key] = l

# Criar um dataset binario, que mostra se a musica está contida ou nao naquela playlist ***

#dataset = pd.DataFrame(data)
'''
for i in range(0, 10):
	print(i)
	print(data[i])
	print()
exit(-1)
'''
'''
data_ = [['song1', 'song2', 'song3'],
        ['song1', 'song4'],
        ['song2', 'song5'],
        ['song6', 'song7']]

print(data[0])
print(data_[0])
'''

# Transaction encoding
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
print(te_ary[0])
exit(-1)

df = pd.DataFrame(te_ary, columns=te.columns_)

# FP-Growth
frequent_itemsets = fpgrowth(df, min_support=0.03, use_colnames=True)

# Association Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
#print(rules)
#print('\n\n')

# User input
#user_input = ['Ride Wit Me', 'Fight Night', 'I Got the Keys'] # songs from different playlists
#user_input = ['Ride Wit Me', 'Red Solo Cup', 'Sweet Emotion'] # songs from the same playlist
#user_input = ['Ride Wit Me', 'Fight Night', 'John Cougar, John Deere, John 3:16']
#user_input = ['Ride Wit Me', 'Fight Night']
#user_input = ['Ride Wit Me', 'Red Solo Cup']
user_input = ['Ride Wit Me']

# User input
#user_input = ['song4']

# Filter rules for relevant recommendations
user_rules = rules[rules['antecedents'].apply(lambda x: set(user_input).issubset(set(x)))]

# Convert frozensets to sets and extract recommended songs
recommended_songs = set().union(*(map(set, user_rules['consequents'])))
print(f"\nRecommended songs: {recommended_songs}")
print(f"# Recommended songs: {len(recommended_songs)}")
print()

if recommended_songs == set():
	print("Empty recommended_songs")
	exit(-1)

# Map row indices to playlists
playlist_mapping = {i: playlist for i, playlist in enumerate(data)}

# Correlate recommended songs with playlists
#correlated_playlists = {i: playlist_mapping[i] for i in range(len(data)) if set(recommended_songs).intersection(set(data[i]))}
correlated_playlists = {i: playlist_mapping[i] for i in range(len(data)) if set(recommended_songs).issubset(set(data[i]))}

#print("Correlated Playlists: ", correlated_playlists)
print(f"Playlists lines: {len(correlated_playlists)}\n")

# Translate pids of dataset to real pids in .csv
playlists_correct = []
for key in correlated_playlists.keys():
	playlists_correct.append(pids_list[key])

#print(f"Correlated Playlists: {playlists_correct}")
# print(f"Correlated Playlists length: {len(playlists_correct)}")

print(f"Correlated Playlists length uniques: {len(set(playlists_correct))}")
print(f"Correlated Playlists unique sorted: {sorted(set(playlists_correct), key=int)}")

#print(len(playlist_mapping.keys()))