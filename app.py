from flask import Flask, render_template, request, jsonify
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date

from techniques.vader import aggregate_daily_sentiment_vader
from techniques.svm import aggregate_daily_sentiment_svm
from techniques.lstm import aggregate_daily_sentiment_lstm
from techniques.finbert import aggregate_daily_sentiment_finbert


app = Flask(__name__)

def get_next_trading_days(start_date, days_ahead=2):
    if hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
        start_date = start_date.replace(tzinfo=None)
    
    if isinstance(start_date, date) and not isinstance(start_date, datetime):
        start_date = datetime.combine(start_date, datetime.min.time())

    trading_days = []
    current_date = start_date + timedelta(days=1)  
    
    while len(trading_days) < days_ahead and len(trading_days) < 10:  
        if current_date.weekday() < 5:  
            trading_days.append(current_date.date())
        current_date += timedelta(days=1)
    
    return trading_days

def get_stock_price_data(ticker_symbol, start_date, end_date):
    try:
        clean_ticker = ticker_symbol.replace('$', '')
        
        if hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
            start_date = start_date.replace(tzinfo=None)
        if hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
            end_date = end_date.replace(tzinfo=None)
        
        stock = yf.Ticker(clean_ticker)
        stock_price_history = stock.history(start=start_date, end=end_date)
        
        if stock_price_history.empty:
            return None
            
        return stock_price_history
    except Exception as e:
        print(f"Error getting stock data for {ticker_symbol}: {e}")
        return None

def get_baseline_price(sentiment_date, stock_data):
    sentiment_date_str = sentiment_date.strftime('%Y-%m-%d')
    
    if sentiment_date.weekday() >= 5: # is a weekend  
        if sentiment_date.weekday() == 5: # saturday  
            previous_friday = sentiment_date - timedelta(days=1)
        else:  # Sunday 
            previous_friday = sentiment_date - timedelta(days=2)
        
        previous_friday_str = previous_friday.strftime('%Y-%m-%d')
        
        try:
            return stock_data.loc[previous_friday_str]['Close']
        except KeyError:
            # if previous Friday not available look up most recent closing price - in case of holidays
            for i in range(1, 8):  
                check_date = sentiment_date - timedelta(days=i)
                if check_date.weekday() < 5:  # Found a weekday
                    check_date_str = check_date.strftime('%Y-%m-%d')
                    try:
                        return stock_data.loc[check_date_str]['Close']
                    except KeyError:
                        continue
            return None
    else:
        try:
            return stock_data.loc[sentiment_date_str]['Close']
        except KeyError:
            return None

def evaluate_sentiment_predictions(daily_sentiment, stock_ticker):
    print(f"Starting evaluation for {stock_ticker} with {len(daily_sentiment)} daily sentiment entries")
    
    evaluation_results = []
    weekend_count = 0
    weekday_count = 0
    
    for sentiment_entry in daily_sentiment:
        sentiment_date = sentiment_entry['date']
        sentiment_prediction = sentiment_entry['sentiment_prediction']
        
        if sentiment_date.weekday() >= 5:
            weekend_count += 1
        else:
            weekday_count += 1
        
        next_trading_days = get_next_trading_days(sentiment_date, days_ahead=2)
        
        if len(next_trading_days) < 2:
            print(f"  Skipping {sentiment_date}: insufficient trading days")
            continue  # skip if not enough trading days i.e. end of the dataset
            
        start_date = sentiment_date
        end_date = next_trading_days[1] + timedelta(days=1)
        
        if sentiment_date.weekday() >= 5:  # Weekend
            if sentiment_date.weekday() == 5:  # Saturday
                start_date = sentiment_date - timedelta(days=3) # ensures we fetch enough data from yfinance in order to conduct evaluation
            else:  # Sunday
                start_date = sentiment_date - timedelta(days=4) # ensures we fetch enough data from yfinance in order to conduct evaluation
        
        if hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
            start_date = start_date.replace(tzinfo=None)
        if hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
            end_date = end_date.replace(tzinfo=None)
        
        if stock_ticker == "$FB": # FB is the ticker for Meta in the datasets used
            stock_ticker = "META"
        
        stock_data = get_stock_price_data(stock_ticker, start_date, end_date)
        
        if stock_data is None or len(stock_data) < 2:
            print(f"  Skipping {sentiment_date}: no stock data available")
            continue
            
        try:
            baseline_price = get_baseline_price(sentiment_date, stock_data)
            if baseline_price is None:
                print(f"  Skipping {sentiment_date}: no baseline price available")
                continue
                
            future_date_str = next_trading_days[1].strftime('%Y-%m-%d')
            future_date_price = stock_data.loc[future_date_str]['Close']
            
            price_change = future_date_price - baseline_price
            price_change_percent = (price_change / baseline_price) * 100
            
            if price_change > 0:
                actual_movement = 'increase'
            elif price_change < 0:
                actual_movement = 'decrease'
            else:
                actual_movement = 'neutral'
                
            prediction_correct = (sentiment_prediction == actual_movement)
            
            if sentiment_date.weekday() >= 5:  # Weekend
                if sentiment_date.weekday() == 5:  # Saturday
                    baseline_date = sentiment_date - timedelta(days=1)  # Friday
                else:  # Sunday
                    baseline_date = sentiment_date - timedelta(days=2)  # Friday
            else:
                baseline_date = sentiment_date
            
            evaluation_results.append({
                'sentiment_date': sentiment_date,
                'sentiment_prediction': sentiment_prediction,
                'sentiment_score': sentiment_entry['average_sentiment'],
                'baseline_date': baseline_date,
                'future_trading_date': next_trading_days[1],
                'baseline_price': baseline_price,
                'future_date_price': future_date_price,
                'price_change': price_change,
                'price_change_percent': price_change_percent,
                'actual_movement': actual_movement,
                'prediction_correct': prediction_correct,
                'is_weekend_sentiment': sentiment_date.weekday() >= 5
            })            
            
        except (IndexError, KeyError) as e:
            print(f"  Error evaluating {sentiment_date}: {e}")
            continue
    
    return evaluation_results

def calculate_performance_metrics(evaluation_results):
    if not evaluation_results:
        print("No evaluation results to calculate metrics from")
        return {}
    
    binary_results = [r for r in evaluation_results if r['sentiment_prediction'] != 'neutral']
        
    if not binary_results:
        print("No binary predictions available for metrics calculation")
        return {'error': 'No binary predictions available for metrics calculation'}
    
    # Calculate confusion matrix
    tp = sum(1 for r in binary_results if r['prediction_correct'] and r['sentiment_prediction'] == 'increase')
    tn = sum(1 for r in binary_results if r['prediction_correct'] and r['sentiment_prediction'] == 'decrease')
    fp = sum(1 for r in binary_results if not r['prediction_correct'] and r['sentiment_prediction'] == 'increase')
    fn = sum(1 for r in binary_results if not r['prediction_correct'] and r['sentiment_prediction'] == 'decrease')
        
    # Calculate metrics
    accuracy = (tp + tn) / len(binary_results) if len(binary_results) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    sentiment_scores = [r['sentiment_score'] for r in evaluation_results]
    price_changes = [r['price_change_percent'] for r in evaluation_results]
    
    if len(sentiment_scores) > 1:
        correlation = pd.Series(sentiment_scores).corr(pd.Series(price_changes))
    else:
        correlation = 0
    
    result = {
        'binary_predictions': len(binary_results),
        'trading_days_evaluated': len(evaluation_results),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'pearson_correlation': correlation,
        'confusion_matrix': {
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
    }
    
    return result

def create_sentiment_vs_price_chart(evaluation_results, stock_ticker, technique):
    if not evaluation_results:
        return None
    
    dates = [pd.to_datetime(r['sentiment_date']) for r in evaluation_results]
    sentiment_scores = [r['sentiment_score'] for r in evaluation_results]
    price_changes = [r['price_change_percent'] for r in evaluation_results]
    predictions = [r['sentiment_prediction'] for r in evaluation_results]
    is_weekend = [r['is_weekend_sentiment'] for r in evaluation_results]
    
    colors = []
    for pred in predictions:
        if pred == 'increase':
            colors.append('green')
        elif pred == 'decrease':
            colors.append('red')
        else:
            colors.append('gray')
    
    marker_symbols = ['circle' if not weekend else 'diamond' for weekend in is_weekend]
    
    fig = {
        'data': [{
            'x': sentiment_scores,
            'y': price_changes,
            'mode': 'markers',
            'type': 'scatter',
            'marker': {
                'color': colors,
                'size': 8,
                'opacity': 0.7,
                'symbol': marker_symbols
            },
            'text': [f"Date: {d.strftime('%d/%m/%Y')}<br>Prediction: {p}<br>Price Change: {pc:.2f}%<br>Weekend: {'Yes' if w else 'No'}" 
                     for d, p, pc, w in zip(dates, predictions, price_changes, is_weekend)],
            'hovertemplate': '%{text}<extra></extra>',
            'name': 'Sentiment vs Price Change'
        }],
        'layout': {
            'title': f'{stock_ticker} {technique} Sentiment Scores vs Price Changes (48h)',
            'height': 600,
            'width': 1200,
            'xaxis': {'title': f'{technique} Sentiment Score'},
            'yaxis': {'title': 'Price Change (%)'},
            'hovermode': 'closest',
            'showlegend': False,
            'autosize': True,
            'annotations': [
                {
                    'text': 'Circle = Weekday, Diamond = Weekend',
                    'showarrow': False,
                    'x': 0.02,
                    'y': 0.98,
                    'xref': 'paper',
                    'yref': 'paper',
                    'bgcolor': 'rgba(255,255,255,0.8)',
                    'bordercolor': 'black',
                    'borderwidth': 1
                }
            ]
        }
    }
    
    return fig

def create_time_series_chart(evaluation_results, stock_ticker, technique):
    if not evaluation_results:
        return None

    dates = [pd.to_datetime(r['sentiment_date']) for r in evaluation_results]
    sentiment_scores = [r['sentiment_score'] for r in evaluation_results]
    price_changes = [r['price_change_percent'] for r in evaluation_results]
    predictions = [r['sentiment_prediction'] for r in evaluation_results]
    is_weekend = [r['is_weekend_sentiment'] for r in evaluation_results]
    
    fig = {
        'data': [
            {
                'x': dates,
                'y': sentiment_scores,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': f'{technique} Sentiment Score',
                'yaxis': 'y',
                'line': {'color': 'blue'},
                'marker': {
                    'size': 6,
                    'symbol': ['diamond' if w else 'circle' for w in is_weekend]
                },
            },
            {
                'x': dates,
                'y': price_changes,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Price Change (%)',
                'yaxis': 'y2',
                'line': {'color': 'orange'},
                'marker': {
                    'size': 6,
                    'symbol': ['diamond' if w else 'circle' for w in is_weekend]
                },
            }
        ],
        'layout': {
            'title': f'{stock_ticker} {technique} Sentiment vs Price Changes Over Time',
            'height': 800,
            'width': 1200,
            'autosize': True,
            'xaxis': {
                        'title': 'Date', 
                        'automargin': True,
                        'tickformat': "%Y-%m-%d",  
                        'hoverformat': "%Y-%m-%d",
                        'tickangle': 90,        
                      },
            'yaxis': {
                'title': f'{technique} Sentiment Score',
                'side': 'left',
                'range': [-1, 1]
            },
            'yaxis2': {
                'title': 'Price Change (%)',
                'side': 'right',
                'overlaying': 'y',
                'range': [-20, 20]
            },
            'hovermode': 'x unified',
            'legend': {'x': 0.02, 'y': 0.98},
            'annotations': [
                {
                    'text': 'Circle = Weekday, Diamond = Weekend',
                    'showarrow': False,
                    'x': 0.02,
                    'y': 0.02,
                    'xref': 'paper',
                    'yref': 'paper',
                    'bgcolor': 'rgba(255,255,255,0.8)',
                    'bordercolor': 'black',
                    'borderwidth': 1
                }
            ]
        }
    }
    
    return fig


def create_performance_summary_chart(performance_metrics):
    if not performance_metrics or 'error' in performance_metrics:
        return None
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [
        performance_metrics['accuracy'],
        performance_metrics['precision'],
        performance_metrics['recall'],
        performance_metrics['f1_score']
    ]
    
    # Create color mapping
    colors = ['green' if v > 0.5 else 'orange' if v > 0.3 else 'red' for v in values]
    
    fig = {
        'data': [{
            'x': metrics,
            'y': values,
            'type': 'bar',
            'marker': {'color': colors},
            'text': [f'{v:.3f}' for v in values],
            'textposition': 'auto'
        }],
        'layout': {
            'title': 'Model Performance Metrics',
            'height': 600,
            'width': 1200,
            'autosize': True,
            'yaxis': {
                'title': 'Score',
                'range': [0, 1]
            },
            'showlegend': False
        }
    }
    
    return fig

def load_dataset(dataset_name):
    if dataset_name == "dataset1":
        return pd.read_csv('data/tweets_labelled_final.csv')
    elif dataset_name == "dataset2":
        return pd.read_csv('data/Full_tokenized_DNN_final.csv')
    elif dataset_name == "both":
        df1 = pd.read_csv('data/tweets_labelled_final.csv')
        df2 = pd.read_csv('data/Full_tokenized_DNN_final.csv')
        
        df1['source_dataset'] = 'dataset1'
        df2['source_dataset'] = 'dataset2'
        
        combined_df = pd.concat([df1, df2], ignore_index=True)
        print(f"Combined datasets: {len(df1)} from dataset1, {len(df2)} from dataset2, total: {len(combined_df)}")
        
        return combined_df
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def contains_ticker(ticker_str, target_ticker):
    if pd.isna(ticker_str):
        return False
    try:
        ticker_list = ticker_str.strip('[]').replace("'", "").replace('"', '').split(', ')
        return target_ticker in ticker_list
    except:
        return False

def filter_tweets_by_ticker(df, stock_ticker):
    ticker_symbol = stock_ticker.replace('$', '')    
    filtered_df = df[df['tickers'].apply(lambda x: contains_ticker(x, ticker_symbol))]
    return filtered_df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    stock_ticker = request.form.get('stock_ticker')
    dataset = request.form.get('dataset')
    technique = request.form.get('technique')
    
    print(f"Analysis request: {technique} on {dataset} for {stock_ticker}")
    
    try:
        df = load_dataset(dataset)
        print(f"Loaded dataset with {len(df)} tweets")
        
        dataset_breakdown = None
        if dataset == "both":
            dataset_breakdown = {
                'dataset1': len(df[df['source_dataset'] == 'dataset1']),
                'dataset2': len(df[df['source_dataset'] == 'dataset2'])
            }
        
        filtered_df = filter_tweets_by_ticker(df, stock_ticker)
        
        filtered_breakdown = None
        if dataset == "both":
            filtered_breakdown = {
                'dataset1': len(filtered_df[filtered_df['source_dataset'] == 'dataset1']),
                'dataset2': len(filtered_df[filtered_df['source_dataset'] == 'dataset2'])
            }
        
        if technique == "VADER":
            daily_sentiment = aggregate_daily_sentiment_vader(filtered_df, stock_ticker)
        elif technique == "SVM":
            daily_sentiment = aggregate_daily_sentiment_svm(filtered_df, stock_ticker, dataset)
        elif technique == "RNN (LSTM)":
            daily_sentiment = aggregate_daily_sentiment_lstm(filtered_df, stock_ticker, dataset)
        elif technique == "FinBERT":
            daily_sentiment = aggregate_daily_sentiment_finbert(filtered_df, stock_ticker)
        else:
            return jsonify({
                'message': f'{technique} analysis not yet implemented',
                'stock_ticker': stock_ticker,
                'dataset': dataset,
                'technique': technique,
                'total_tweets': len(df),
                'filtered_tweets': len(filtered_df)
            })
        
        evaluation_results = evaluate_sentiment_predictions(daily_sentiment, stock_ticker)
        performance_metrics = calculate_performance_metrics(evaluation_results)
        sentiment_vs_price_chart = create_sentiment_vs_price_chart(evaluation_results, stock_ticker, technique)
        time_series_chart = create_time_series_chart(evaluation_results, stock_ticker, technique)
        performance_chart = create_performance_summary_chart(performance_metrics)
        
        return jsonify({
            'message': 'Daily sentiment analysis completed successfully!',
            'stock_ticker': stock_ticker,
            'dataset': dataset,
            'technique': technique,
            'total_tweets': len(df),
            'filtered_tweets': len(filtered_df),
            'dataset_breakdown': dataset_breakdown,
            'filtered_breakdown': filtered_breakdown,
            'daily_sentiment': daily_sentiment,
            'analysis_type': 'daily_aggregation_for_stock_prediction',
            'performance_metrics': performance_metrics,
            'evaluation_results': evaluation_results,
            'charts': {
                'sentiment_vs_price': sentiment_vs_price_chart,
                'time_series': time_series_chart,
                'performance_summary': performance_chart
            }
        })
   
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': f'Error processing request: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000) 