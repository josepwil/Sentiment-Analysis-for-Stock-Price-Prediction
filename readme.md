# Sentiment Analysis for Stock Price Prediction

A tool to allow the comparative study of sentiment analysis techniques (VADER, SVM, LSTM, FinBERT) for predicting stock price movements using Twitter data.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

3. **Open your browser:**
   Navigate to `http://localhost:5000`

## Usage

1. Select a stock ticker (e.g., AAPL, MSFT, GOOGL)
2. Choose a dataset (dataset1, dataset2, or both)
3. Select a sentiment analysis technique
4. Click "Analyze" to view results and performance metrics

## Data

The project uses pre-processed Twitter datasets with sentiment labels. Ensure the following CSV files are in the project directory:
- `tweets_labelled_final.csv`
- `Full_tokenized_DNN_final.csv`
- `tweets_labelled_final_train.csv`
- `tweets_labelled_final_test.csv`
- `Full_tokenized_DNN_final_train.csv`
- `Full_tokenized_DNN_final_test.csv`

## Results

The application generates:
- Sentiment vs. price change scatter plots
- Time series analysis charts
- Performance metrics (accuracy, precision, recall, F1-score)
- Correlation analysis between sentiment and stock movements

Pre-computed results for all technique/dataset combinations are available in the `results/` folder.

Each folder contains:
- Performance metrics charts
- Sentiment vs. price correlation plots
- Time series analysis visualizations
- Stock-specific results (AAPL, AMZN, FB, GS)

**Note**: Results are organized by technique and include both individual dataset results and combined dataset results.