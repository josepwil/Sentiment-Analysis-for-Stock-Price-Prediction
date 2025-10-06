from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

vader_analyzer = SentimentIntensityAnalyzer()

def aggregate_daily_sentiment_vader(tweets_df, stock_ticker):
    tweets_df = tweets_df.copy()
    
    if 'datetime' not in tweets_df.columns:
        tweets_df['datetime'] = pd.to_datetime(tweets_df['date'])
    
    tweets_df['datetime'] = pd.to_datetime(tweets_df['datetime']).dt.tz_localize(None)
    tweets_df['date'] = tweets_df['datetime'].dt.date
    
    daily_sentiment = []
    
    for date, group in tweets_df.groupby('date'):
        date_sentiments = []
        
        for _, row in group.iterrows():
            text = row.get('normalized_text', row.get('TokenizedText', ''))
            
            sentiment_scores = vader_analyzer.polarity_scores(text)
            date_sentiments.append(sentiment_scores['compound'])
        
        if date_sentiments:
            daily_result = {
                'date': date,
                'stock_ticker': stock_ticker,
                'tweet_count': len(date_sentiments),
                'average_sentiment': sum(date_sentiments) / len(date_sentiments),
                'sentiment_prediction': 'increase' if sum(date_sentiments) / len(date_sentiments) > 0.05 else 
                                      'decrease' if sum(date_sentiments) / len(date_sentiments) < -0.05 else 'neutral'
            }
            daily_sentiment.append(daily_result)
    
    daily_sentiment.sort(key=lambda x: x['date'])
    
    return daily_sentiment
