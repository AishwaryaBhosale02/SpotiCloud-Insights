import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from wordcloud import WordCloud

# Initialize the sentiment analysis pipeline
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def analyze_sentiment_and_generate_wordclouds(df):
    # Analyze sentiment for each review
    sentiments = []
    for review in df['Review']:
        result = sentiment_pipeline(review)
        sentiment_label = result[0]['label']
        
        # Convert model's output to Positive, Negative, or Neutral
        if sentiment_label in ['1 star', '2 stars']:
            sentiments.append('Negative')
        elif sentiment_label in ['4 stars', '5 stars']:
            sentiments.append('Positive')
        else:
            sentiments.append('Neutral')
    
    df['Sentiment'] = sentiments

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
