import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud

def analyze_sentiment_and_generate_wordclouds(df):
    # Initialize VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Analyze sentiment for each review
    sentiments = []
    compound_scores = []
    for review in df['Review']:
        sentiment_scores = sid.polarity_scores(review)
        if sentiment_scores['compound'] >= 0.05:
            sentiment = 'Positive'
        elif sentiment_scores['compound'] <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        sentiments.append(sentiment)
        compound_scores.append(sentiment_scores['compound'])

    # Add sentiment and compound score columns to dataframe
    df['Sentiment'] = sentiments
    df['Compound_Score'] = compound_scores

    # Generate word clouds for positive, negative, and neutral reviews
    positive_reviews = " ".join(df[df['Sentiment'] == 'Positive']['Review'])
    negative_reviews = " ".join(df[df['Sentiment'] == 'Negative']['Review'])
    neutral_reviews = " ".join(df[df['Sentiment'] == 'Neutral']['Review'])

    # Initialize paths to word cloud images
    wordcloud_positive_path = 'static/wordcloud_positive.png'
    wordcloud_negative_path = 'static/wordcloud_negative.png'
    wordcloud_neutral_path = 'static/wordcloud_neutral.png'

    # Generate and save word clouds if reviews are available
    if positive_reviews:
        wordcloud_positive = WordCloud(width=600, height=300, background_color='white').generate(positive_reviews)
        wordcloud_positive.to_file(wordcloud_positive_path)
    else:
        wordcloud_positive_path = None

    if negative_reviews:
        wordcloud_negative = WordCloud(width=600, height=300, background_color='white').generate(negative_reviews)
        wordcloud_negative.to_file(wordcloud_negative_path)
    else:
        wordcloud_negative_path = None

    if neutral_reviews:
        wordcloud_neutral = WordCloud(width=600, height=300, background_color='white').generate(neutral_reviews)
        wordcloud_neutral.to_file(wordcloud_neutral_path)
    else:
        wordcloud_neutral_path = None

    return df, wordcloud_positive_path, wordcloud_negative_path, wordcloud_neutral_path
