import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train_svm_model(dataset_selection):  
    try:
        if dataset_selection == "dataset1":
            train_df = pd.read_csv('data/tweets_labelled_final_train.csv')
            test_df = pd.read_csv('data/tweets_labelled_final_test.csv')
        elif dataset_selection == "dataset2":
            train_df = pd.read_csv('data/Full_tokenized_DNN_final_train.csv')
            test_df = pd.read_csv('data/Full_tokenized_DNN_final_test.csv')
        elif dataset_selection == "both":
            train_df1 = pd.read_csv('data/tweets_labelled_final_train.csv')
            train_df2 = pd.read_csv('data/Full_tokenized_DNN_final_train.csv')
            test_df1 = pd.read_csv('data/tweets_labelled_final_test.csv')
            test_df2 = pd.read_csv('data/Full_tokenized_DNN_final_test.csv')
            
            train_df = pd.concat([train_df1, train_df2], ignore_index=True)
            test_df = pd.concat([test_df1, test_df2], ignore_index=True)
        else:
            raise ValueError(f"Unknown dataset selection: {dataset_selection}")
        
        print(f"Training on {len(train_df)} samples, testing on {len(test_df)} samples")
        
        X_train = train_df['normalized_text'].fillna('')
        y_train = train_df['sentiment']
        
        X_test = test_df['normalized_text'].fillna('')
        y_test = test_df['sentiment']
        
        train_mask = y_train.notna()
        test_mask = y_test.notna()
        
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        
        print(f"Training on {len(X_train)} valid samples, testing on {len(X_test)} valid samples")
        print(f"Training sentiment distribution: {y_train.value_counts().sort_index().to_dict()}")

        # add weighting due to class imbalance
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        print(f"Class weights: {class_weight_dict}")
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', min_df=2)),
            ('svm', SVC(kernel='linear', probability=True, random_state=42, class_weight=class_weight_dict))
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"SVM model trained with {accuracy:.3f} accuracy on test set")
        
        return pipeline
        
    except Exception as e:
        print(f"Error training SVM model for {dataset_selection}: {e}")
        return None

def aggregate_daily_sentiment_svm(tweets_df, stock_ticker, dataset_selection):
    tweets_df = tweets_df.copy()
    
    if 'datetime' not in tweets_df.columns:
        tweets_df['datetime'] = pd.to_datetime(tweets_df['date'])
    
    tweets_df['datetime'] = pd.to_datetime(tweets_df['datetime']).dt.tz_localize(None)
    tweets_df['date'] = tweets_df['datetime'].dt.date
    
    svm_model = train_svm_model(dataset_selection)
    if svm_model is None:
        print(f"Error: Could not load SVM model for {dataset_selection}")
        return []
    
    daily_sentiment = []
    
    for date, group in tweets_df.groupby('date'):
        date_sentiments = []
        
        for _, row in group.iterrows():
            text = row.get('normalized_text', row.get('TokenizedText', ''))
            
            if text and text.strip():
                try:
                    prediction = svm_model.predict([text])[0]
                    probabilities = svm_model.predict_proba([text])[0]
                    
                    if prediction == 1:  # positive
                        sentiment_score = probabilities[1]
                    elif prediction == -1:  # negative
                        sentiment_score = -probabilities[0]  
                    else:  # This shouldn't happen with but just in case
                        sentiment_score = 0.0
                    
                    date_sentiments.append(sentiment_score)
                except Exception as e:
                    print(f"Error processing text with SVM: {e}")
                    continue
        
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
