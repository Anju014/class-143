from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Remember, we do not have to do any sort of processing. 
# We have already exported this CSV from google colab,
# which means that it already contains the metadata soup string that we created. 

# One thing that we need to ensure is to drop the rows that do not
# have a valid metadata soup string.


df = pd.read_csv('final.csv')
df = df[df['soup'].notna()]

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

df = df.reset_index()
indices = pd.Series(df.index, index=df['title'])

def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df[['title', 'poster_link', 'release_date', 'runtime', 'vote_average', 'overview']].iloc[movie_indices].values.tolist()