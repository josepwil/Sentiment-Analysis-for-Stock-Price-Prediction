import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_finbert_model():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_safetensors=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        use_safetensors=True,   
        torch_dtype=torch.float32
    )
    model.eval()

    return model, tokenizer


def aggregate_daily_sentiment_finbert(tweets_df, stock_ticker):    
    tweets_df = tweets_df.copy()
    
    if 'datetime' not in tweets_df.columns:
        tweets_df['datetime'] = pd.to_datetime(tweets_df['date'])
    
    tweets_df['datetime'] = pd.to_datetime(tweets_df['datetime']).dt.tz_localize(None)
    tweets_df['date'] = tweets_df['datetime'].dt.date
    
    finbert_model, tokenizer = load_finbert_model()
    if finbert_model is None or tokenizer is None:
        print("Error: Could not load FinBERT model")
        return []
    
    daily_sentiment = []
    
    for date, group in tweets_df.groupby('date'):
        texts = group['normalized_text'].fillna('').tolist()
        valid_texts = [text for text in texts if text.strip()]
        
        if not valid_texts:
            continue
            
        try:
            date_sentiments = []
            batch_size = 8

            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i + batch_size]
                
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                with torch.no_grad():
                    outputs = finbert_model(**inputs)
                    probabilities = F.softmax(outputs.logits, dim=1)
                
                for probs in probabilities:
                    # FinBERT: [negative, neutral, positive]
                    negative_prob = probs[0].item()
                    positive_prob = probs[2].item()
                    sentiment_score = positive_prob - negative_prob
                    date_sentiments.append(sentiment_score)
                    
        except Exception as e:
            print(f"Error processing batch for date {date}: {e}")
            continue
        
        if date_sentiments: 
            avg_sentiment = sum(date_sentiments) / len(date_sentiments)
            daily_result = {
                'date': date,
                'stock_ticker': stock_ticker,
                'tweet_count': len(date_sentiments),
                'average_sentiment': avg_sentiment,
                'sentiment_prediction': (
                    'increase' if avg_sentiment > 0.05 else
                    'decrease' if avg_sentiment < -0.05 else
                    'neutral'
                )
            }
            daily_sentiment.append(daily_result)

    daily_sentiment.sort(key=lambda x: x['date'])
    
    return daily_sentiment
