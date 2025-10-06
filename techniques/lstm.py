import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np



def train_lstm_model(dataset_selection):
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
        y_train = train_df['sentiment'].map({-1.0: 0, 1.0: 1})
        
        X_test = test_df['normalized_text'].fillna('')
        y_test = test_df['sentiment'].map({-1.0: 0, 1.0: 1})
        
        train_mask = y_train.notna()
        test_mask = y_test.notna()
        
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        
        print(f"Training on {len(X_train)} valid samples, testing on {len(X_test)} valid samples")
        print(f"Training sentiment distribution: {y_train.value_counts().sort_index().to_dict()}")
        
        tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
        tokenizer.fit_on_texts(X_train)
        
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        
        max_length = 100
        X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
        X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')
        
        # use class weights due to class imbalance
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        print(f"Class weights: {class_weight_dict}")
        
        model = Sequential([
            Embedding(10000, 128, input_length=max_length),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(32, return_sequences=False)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')  # Binary output with sigmoid
        ])
        
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        

        print("LSTM Model Architecture:")
        model.summary()
        

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        
        history = model.fit(
            X_train_padded, y_train,  
            epochs=25,  
            batch_size=32,
            validation_data=(X_test_padded, y_test),
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weight_dict,
            verbose=1
        )
        
        print(f"Training completed. Final accuracy: {history.history['accuracy'][-1]:.3f}")
        
        test_loss, test_accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
        print(f"LSTM model trained with {test_accuracy:.3f} accuracy on test set")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error training LSTM model for {dataset_selection}: {e}")
        return None, None

def aggregate_daily_sentiment_lstm(tweets_df, stock_ticker, dataset_selection):    
    tweets_df = tweets_df.copy()
    
    if 'datetime' not in tweets_df.columns:
        tweets_df['datetime'] = pd.to_datetime(tweets_df['date'])
    
    tweets_df['datetime'] = pd.to_datetime(tweets_df['datetime']).dt.tz_localize(None)
    tweets_df['date'] = tweets_df['datetime'].dt.date

    lstm_model, tokenizer = train_lstm_model(dataset_selection)
    if lstm_model is None:
        print("LSTM model training failed")
        return jsonify({
            'error': 'Could not train LSTM model. Please check logs for details.'
        }), 500
    
    daily_sentiment = []
    
    for date, group in tweets_df.groupby('date'):
        texts = group['normalized_text'].fillna('').tolist()
        valid_texts = [text for text in texts if text and text.strip()]
        
        if not valid_texts:
            continue
            
        try:
            text_seqs = tokenizer.texts_to_sequences(valid_texts)
            text_padded = pad_sequences(text_seqs, maxlen=100, padding='post', truncating='post')
            
            batch_predictions = lstm_model.predict(text_padded, verbose=0)
            
            date_sentiments = []
            for i, prediction_probs in enumerate(batch_predictions):
                prediction_prob = prediction_probs[0]
                
                # use a lower threshold for positive prediction to handle class imbalance
                prediction = 1 if prediction_prob > 0.35 else 0  
                                
                # convert prediction to sentiment score (-1 to 1 scale)
                sentiment_score = (prediction_prob - 0.5) * 2
                
                date_sentiments.append(sentiment_score)
                
        except Exception as e:
            print(f"Error processing batch for date {date}: {e}")
            continue
        
        if date_sentiments:
            avg_sentiment = sum(date_sentiments) / len(date_sentiments)
            print(f"  Date {date}: average sentiment = {avg_sentiment:.3f}")
            
            daily_result = {
                'date': date,
                'stock_ticker': stock_ticker,
                'tweet_count': len(date_sentiments),
                'average_sentiment': avg_sentiment,
                'sentiment_prediction': 'increase' if avg_sentiment > 0.01 else 
                                      'decrease' if avg_sentiment < -0.01 else 'neutral'
            }
            daily_sentiment.append(daily_result)
    
    daily_sentiment.sort(key=lambda x: x['date'])
    
    print(f"LSTM generated daily sentiment for {len(daily_sentiment)} days")
    if daily_sentiment:
        print(f"Sample sentiment scores: {[round(d['average_sentiment'], 3) for d in daily_sentiment[:5]]}")
    
    return daily_sentiment
