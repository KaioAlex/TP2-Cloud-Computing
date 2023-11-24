from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import csv
import pandas as pd
import pickle
from datetime import datetime

from itertools import groupby
from operator import itemgetter

import wget
import os.path

if not os.path.isfile("2023_spotify_ds1.csv"):
	filename = wget.download("https://github.com/KaioAlex/TP2-Cloud-Computing/raw/main/datasets/2023_spotify_ds1.csv")

reader = csv.DictReader(open("2023_spotify_ds1.csv"), delimiter=",")
songs_list = []

for row in reader:
	songs_list.append(row)

data = {}
# Display songs grouped by pids
for key, value in groupby(songs_list, key = itemgetter('pid')):
	l = []

	for k in value:
		l.append(k['track_name'])

	data[key] = l

playlists_df = pd.DataFrame(list(data.items()), columns=['pids', 'songs'])

# Combine song names into playlists
playlists_df['combined_songs'] = playlists_df['songs'].apply(lambda x: ' '.join(x))

# Use TF-IDF to vectorize playlist features
tfidf_vectorizer = TfidfVectorizer()
playlist_tfidf_matrix = tfidf_vectorizer.fit_transform(playlists_df['combined_songs'])

version = 0
try:
	file = open("./trained_model.pickle", "rb")
	last_model = pickle.load(file)
	version = last_model['version'] + 1
except:
	pass

persist = {
	"playlists_df": playlists_df,
	"playlist_tfidf_matrix": playlist_tfidf_matrix,
	"tfidf_vectorizer": tfidf_vectorizer,
	"version": version,
	"model_date": datetime.now().isoformat()
}

file = open("./trained_model.pickle", "wb")
pickle.dump(persist, file)
file.close()

print("Trained model saved: 'trained_model.pickle'")