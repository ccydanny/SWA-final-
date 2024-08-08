import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

# Load the dataset
df = pd.read_csv('Recipe Reviews and User Feedback Dataset.csv')

# Data Cleaning
## Remove unwanted columns
df.drop(columns=['Unnamed: 0', 'recipe_number', 'comment_id', 'user_id', 'user_name', 'created_at'], inplace=True)

## Handle missing values
df.fillna('', inplace=True)

## Rename columns (if needed)
df.rename(columns={
    'recipe_code': 'Recipe Code',
    'recipe_name': 'Recipe Name',
    'user_reputation': 'User Reputation',
    'reply_count': 'Reply Count',
    'thumbs_up': 'Thumbs Up',
    'thumbs_down': 'Thumbs Down',
    'stars': 'Stars',
    'best_score': 'Best Score',
    'text': 'Review Text'
}, inplace=True)

# Ensure all Review Text entries are strings
df['Review Text'] = df['Review Text'].astype(str)

# Text Preprocessing
## Convert to lowercase
df['Review Text'] = df['Review Text'].str.lower()

## Remove punctuation
df['Review Text'] = df['Review Text'].str.translate(str.maketrans('', '', string.punctuation))

## Remove newline characters
df['Review Text'] = df['Review Text'].str.replace('\n', ' ').str.replace('\r', '')

## Remove non-alphabetic characters (including emojis)
df['Review Text'] = df['Review Text'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))

## Remove stopwords
stop_words = set(stopwords.words('english'))
df['Review Text'] = df['Review Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

## Remove digits
df['Review Text'] = df['Review Text'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))

## Tokenize the review text
df['Review Text Tokenized'] = df['Review Text'].apply(word_tokenize)

## Apply stemming
stemmer = PorterStemmer()
df['Processed Review Text'] = df['Review Text Tokenized'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x]))

# Save cleaned dataset
df.to_csv('cleaned_recipe_dataset.csv', index=False)

# Reload the dataset
df = pd.read_csv('cleaned_recipe_dataset.csv')

# Data cleaning
df['Review Text'] = df['Review Text'].fillna('').astype(str)

# Calculate the mean star rating for each recipe and get the top 10 recipes
top_10_recipes = df.groupby('Recipe Name')['Stars'].mean().nlargest(10).reset_index()

# Filter the dataframe to include only reviews for the top 10 recipes
df_top_10 = df[df['Recipe Name'].isin(top_10_recipes['Recipe Name'])]

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to get sentiment scores
def get_sentiment(text):
    return sia.polarity_scores(text)

# Apply sentiment analysis
df_top_10['sentiment'] = df_top_10['Review Text'].apply(get_sentiment)
df_top_10[['neg', 'neu', 'pos', 'compound']] = df_top_10['sentiment'].apply(pd.Series)

# Generate and plot word clouds
def plot_wordcloud(text, title, background_color='white', colormap=None):
    wordcloud = WordCloud(width=800, height=400, background_color=background_color, colormap=colormap).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Word cloud for positive reviews
positive_reviews = ' '.join(df_top_10[df_top_10['compound'] > 0.2]['Review Text'])
plot_wordcloud(positive_reviews, 'Word Cloud for Positive Reviews (Top 10 Recipes)')

# Word cloud for negative reviews
negative_reviews = ' '.join(df_top_10[df_top_10['compound'] < -0.2]['Review Text'])
plot_wordcloud(negative_reviews, 'Word Cloud for Negative Reviews (Top 10 Recipes)', background_color='black', colormap='Reds')

# Calculate and print average sentiment scores
average_sentiments = df_top_10[['neg', 'neu', 'pos', 'compound']].mean()
print("Average Sentiment Scores:\n", average_sentiments)

# Social Media Action Analytics
correlation_thumb_up = df_top_10[['Thumbs Up', 'compound']].corr().iloc[0, 1]
correlation_thumb_down = df_top_10[['Thumbs Down', 'compound']].corr().iloc[0, 1]

print(f"Correlation between 'Thumbs Up' and sentiment compound score: {correlation_thumb_up:.2f}")
print(f"Correlation between 'Thumbs Down' and sentiment compound score: {correlation_thumb_down:.2f}")

# Scatter plots for Thumbs Up and Thumbs Down
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x='Thumbs Up', y='compound', data=df_top_10)
plt.title("Thumbs Up vs. Sentiment Compound Score")

plt.subplot(1, 2, 2)
sns.scatterplot(x='Thumbs Down', y='compound', data=df_top_10)
plt.title("Thumbs Down vs. Sentiment Compound Score")

plt.tight_layout()
plt.show()

# Plot top 10 recipes by average star rating
plt.figure(figsize=(12, 8))

top_10_recipes = top_10_recipes.sort_values(by='Stars', ascending=True)
sns.barplot(x='Stars', y='Recipe Name', data=top_10_recipes, palette='viridis', edgecolor='black')

# Add annotations to the bar plot
for index, value in enumerate(top_10_recipes['Stars']):
    plt.text(value, index, f'{value:.2f}', va='center', ha='left', color='black', fontsize=10)

# Customize and show the bar plot
plt.xlabel('Average Star Rating', fontsize=14, labelpad=15)
plt.ylabel('Recipe Name', fontsize=14, labelpad=15)
plt.title('Top 10 Recipes by Average Star Rating', fontsize=16, pad=20)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.gca().invert_yaxis()  
plt.xticks(fontsize=12)
plt.yticks(fontsize=7)  

plt.show()
