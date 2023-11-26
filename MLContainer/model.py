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

from github import Github

if not os.path.isfile("training.csv"):
	filename = wget.download("https://github.com/KaioAlex/TP2-Cloud-Computing/raw/main/datasets/training.csv")

reader = csv.DictReader(open("training.csv"), delimiter=",")
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
	filename = wget.download("https://github.com/KaioAlex/TP2-Cloud-Computing/raw/main/MLContainer/trained_model.pickle")
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

# Upload new trained_model.pickle to github

# assign each item to a variable
token = "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"
repo_for_upload = "KaioAlex/TP2-Cloud-Computing"

g = Github(token)

repo = g.get_repo(repo_for_upload)

with open("trained_model.pickle", 'rb') as file:
    data = file.read()

repo.create_file("MLContainer/trained_model.pickle", f"Model Version: {version}", data, branch="main")

if os.path.isfile("training.csv"):
	os.remove("training.csv")