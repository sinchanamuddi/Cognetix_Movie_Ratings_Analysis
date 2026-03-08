import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and Preprocess Datasets [cite: 68]
try:
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
except FileNotFoundError:
    print("Error: Ensure movies.csv and ratings.csv are in the same folder.")

# 2. Merge datasets to associate movies with ratings and genres 
# We use 'movieId' as the common key
data = pd.merge(ratings, movies, on='movieId')

# 3. Calculate average ratings and total review counts 
movie_stats = data.groupby('title').agg({'rating': ['mean', 'count']})
movie_stats.columns = ['Average Rating', 'Review Count']

# 4. Identify top-rated movies with at least 100 reviews (for quality) [cite: 71]
top_rated = movie_stats[movie_stats['Review Count'] > 100].sort_values(by='Average Rating', ascending=False).head(10)
print("Top 10 Rated Movies (min 100 reviews):\n", top_rated)

# 5. Analyze Genre-wise Performance [cite: 72]
# Splitting genres as they are often pipe-separated (e.g., Action|Comedy)
data['genres'] = data['genres'].str.split('|')
exploded_data = data.explode('genres')
genre_analysis = exploded_data.groupby('genres')['rating'].mean().sort_values(ascending=False)

# 6. Visualize Insights 
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_analysis.index, y=genre_analysis.values, palette='viridis')
plt.title('Average User Rating by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7. Visualize Rating Distribution (Histogram) 
plt.figure(figsize=(10, 5))
sns.histplot(ratings['rating'], bins=10, kde=True, color='blue')
plt.title('Distribution of User Ratings')
plt.xlabel('Rating Score')
plt.ylabel('Frequency')
plt.show()