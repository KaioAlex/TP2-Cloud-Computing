from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import csv
import pandas as pd

from itertools import groupby
from operator import itemgetter

# Assuming you have playlists_df and songs_df DataFrames
reader = csv.DictReader(open("./datasets/2023_spotify_ds1.csv"), delimiter=",")
songs_list = []

pids_list = []
for row in reader:
	songs_list.append(row)
	pids_list.append(row['pid'])

data = {}
# Display songs grouped by pids
for key, value in groupby(songs_list, key = itemgetter('pid')):
	l = []

	for k in value:
		l.append(k['track_name'])

	#data.append(l)
	data[key] = l

playlists_df = pd.DataFrame(list(data.items()), columns=['pids', 'songs'])

# Combine song names into playlists
playlists_df['combined_songs'] = playlists_df['songs'].apply(lambda x: ' '.join(x))

# Use TF-IDF to vectorize playlist features
tfidf_vectorizer = TfidfVectorizer()
playlist_tfidf_matrix = tfidf_vectorizer.fit_transform(playlists_df['combined_songs'])

# Vectorize user input songs
user_input_songs = ["Ride Wit Me", "I Got the Keys"]
user_input_vector = tfidf_vectorizer.transform([" ".join(user_input_songs)])

# Calculate cosine similarity
cosine_similarities = linear_kernel(user_input_vector, playlist_tfidf_matrix).flatten()

# Recommend playlists based on similarity scores
#playlist_recommendations = playlists_df.loc[cosine_similarities.argsort()[::-1]]

# Rank playlists based on similarity scores
playlist_ranking = cosine_similarities.argsort()[::-1]

import random

N = random.randint(2, 10)  # You can replace this with any desired value

# Recommend the top-N playlists
top_playlists = playlists_df.iloc[playlist_ranking[:N]]

# Display the recommended playlists
print(f"Top-{N} Recommended Playlists:")
print(top_playlists[['pids']].to_string(index=False))