import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("movies.csv")

# Vectorize genres
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(df['genre'])

# Compute similarity matrix
similarity_matrix = cosine_similarity(genre_matrix)

# Recommend function
def recommend(movie_title, top_n=5):
    if movie_title not in df['title'].values:
        return []
    idx = df[df['title'] == movie_title].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [df.iloc[i[0]]['title'] for i in sorted_scores]

# Styled image output
def show_recommendations(recommendations, base_title):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#f5f5f5')
    ax.axis('off')

    # Title without emoji
    ax.set_title(
        f"Recommended Movies for '{base_title}'",
        fontsize=16, weight='bold', color='#333333', pad=20
    )

    # Format the recommendations text
    content_text = "\n".join([f"{i+1}. {rec}" for i, rec in enumerate(recommendations)])

    # Content styling
    ax.text(
        0.5, 0.5, content_text,
        fontsize=13,
        color="#222222",
        ha='center',
        va='center',
        bbox=dict(boxstyle="round,pad=1", facecolor="#ffffff", edgecolor="#cccccc")
    )

    plt.tight_layout()
    plt.savefig("recommendations.png", dpi=200)
    plt.show()

# Main logic
if __name__ == "__main__":
    print("🎬 Welcome to the Movie Recommendation System!")
    user_input = input("Enter a movie title: ").strip()

    results = recommend(user_input)
    if results:
        print("\nTop Recommendations:")
        for i, movie in enumerate(results, 1):
            print(f"{i}. {movie}")
        show_recommendations(results, user_input)
    else:
        print("❌ Movie not found in the dataset. Please try again.")
