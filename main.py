"""
Recommend items (e.g., movies) that are similar in content to what the user already likes or is viewing.

It focuses on the item's features (e.g., genres, descriptions, actors, etc.), not on user preferences across the system.
"""

import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 1. Load both datasets
movies = pd.read_csv('movies_metadata.csv', low_memory=False)
credits = pd.read_csv('credits.csv')

# 2. Fix data types and merge on id
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
credits['id'] = pd.to_numeric(credits['id'], errors='coerce')

df = movies.merge(credits, on='id')

# 3. Keep only relevant columns and drop missing overviews
df = df[['title', 'overview', 'genres', 'cast', 'crew']].dropna(subset=['overview'])
df = df.head(8000)


# 4. Parse genres, cast, and crew
def parse_genres(genres_str):
    try:
        genres = ast.literal_eval(genres_str)
        return ' '.join([g['name'] for g in genres])
    except:
        return ''


def parse_cast(cast_str, num=3):
    try:
        cast = ast.literal_eval(cast_str)
        return ' '.join([c['name'] for c in cast[:num]])
    except:
        return ''


def parse_crew(crew_str):
    try:
        crew = ast.literal_eval(crew_str)
        for member in crew:
            if member['job'] == 'Director':
                return member['name']
        return ''
    except:
        return ''


df['genres_parsed'] = df['genres'].apply(parse_genres)
df['cast_parsed'] = df['cast'].apply(parse_cast)
df['director'] = df['crew'].apply(parse_crew)

# 5. Combine all text features
df['soup'] = df['overview'] + ' ' + df['genres_parsed'] + ' ' + df['cast_parsed'] + ' ' + df['director']

# 6. TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['soup'])

# 7. Cosine Similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# 8. Map movie titles to indices
indices = pd.Series(df.index, index=df['title']).drop_duplicates()


# 9. Recommendation function
def recommend(title, n=5):
    if title not in indices:
        print("Title not found in dataset.")
        return []

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n + 1]
    rec_indices = [i for i, _ in sim_scores]
    return df[['title', 'overview']].iloc[rec_indices]


# 10. Try an example
if __name__ == "__main__":
    movie = "The Matrix"
    results = recommend(movie)

    print(f"\nTop recommendations for '{movie}':\n")
    for _, row in results.iterrows():
        print(f"{row['title']}: {row['overview'][:150]}...\n")
