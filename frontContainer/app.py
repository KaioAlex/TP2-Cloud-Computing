from flask import Flask, request, jsonify, render_template
import pickle

from sklearn.metrics.pairwise import linear_kernel
import random

app = Flask(__name__)
trained_model = None
project_info = None

@app.route('/')
def home():
    return "<p>Hello, World!</p>"

''' # Renderizar na pagina html
@app.route('/predict', methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))
'''

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json(force=True)

    playlist_tfidf_matrix = trained_model['playlist_tfidf_matrix']
    tfidf_vectorizer = trained_model['tfidf_vectorizer']
    playlists_df = trained_model['playlists_df']

    # Vectorize user input songs
    user_input_songs = data['songs']
    user_input_vector = tfidf_vectorizer.transform([" ".join(user_input_songs)])

    # Calculate cosine similarity
    cosine_similarities = linear_kernel(user_input_vector, playlist_tfidf_matrix).flatten()

    # Rank playlists based on similarity scores
    playlist_ranking = cosine_similarities.argsort()[::-1]

    N = random.randint(2, 10) # random number of playlists between 2 and 10

    # Recommend the top-N playlists
    top_playlists = playlists_df.iloc[playlist_ranking[:N]]

    playlists_recommended = top_playlists[['pids']].to_string(index=False, header=False).split("\n")

    # Display the recommended playlists
    print(f"\n\nBased on the songs: {user_input_songs}")
    print(f"Top-{N} Recommended Playlists:")
    print(f"{playlists_recommended}\n\n")

    return {'playlist_ids': playlists_recommended,
            'version': trained_model['version'],
            'model_date': trained_model['model_date']
    }

def load_model():
  global trained_model
  try:
    file = open("trained_model.pickle", "rb")
    trained_model = pickle.load(file)

    print(f"\nVersion: {trained_model['version']}")
    print(f"Date: {trained_model['model_date']}\n")

    file.close()
  except Exception as error:
      print(f"Could't load trained model: {error}")

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=32174, debug=True)
    