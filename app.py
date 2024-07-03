from flask import Flask, render_template, request
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Import your function from analyze_sentiment.py
from analyze_sentiment import analyze_sentiment_and_generate_wordclouds

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    review = request.form['review']
    
    # Perform sentiment analysis and generate word clouds
    df = pd.DataFrame({'Review': [review]})
    df, wordcloud_positive_path, wordcloud_negative_path, wordcloud_neutral_path = analyze_sentiment_and_generate_wordclouds(df)
    
    # Determine sentiment label based on compound score
    # Since you're no longer calculating compound scores in the function,
    # you can determine sentiment directly from the returned dataframe
    sentiment = df['Sentiment'].iloc[0]
    
    return render_template('results.html', review=review, sentiment=sentiment, 
                           wordcloud_positive_path=wordcloud_positive_path,
                           wordcloud_negative_path=wordcloud_negative_path,
                           wordcloud_neutral_path=wordcloud_neutral_path)

if __name__ == '__main__':
    app.run(debug=True)