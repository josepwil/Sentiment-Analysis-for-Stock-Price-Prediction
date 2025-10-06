import pandas as pd
import re
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split

def normalize_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    
    common_contractions = {
        "don't": "do not",
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",
        "'ll": " will",
        "'re": " are",
        "'ve": " have",
        "'m": " am",
        "'d": " would",
        "'s": " is" 
    }
    
    for contraction, expansion in common_contractions.items():
        text = text.replace(contraction, expansion)
    
    text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def split_data_by_time(df, test_size=0.2, random_state=42):
    df_sorted = df.sort_values('datetime').reset_index(drop=True)

    # Split the data
    split_idx = int(len(df_sorted) * (1 - test_size))
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    print(f"Data split:")
    print(f"  Training set: {len(train_df)} samples ({len(train_df)/len(df_sorted)*100:.1f}%)")
    print(f"  Test set: {len(test_df)} samples ({len(test_df)/len(df_sorted)*100:.1f}%)")
    print(f"  Training date range: {train_df['datetime'].min()} to {train_df['datetime'].max()}")
    print(f"  Test date range: {test_df['datetime'].min()} to {test_df['datetime'].max()}")
    
    return train_df, test_df

def preprocess_for_models(df):
    print("Preprocessing text for models...")
    
    tqdm.pandas(desc="Normalizing text")
    df['normalized_text'] = df['TokenizedText'].progress_apply(normalize_text)
    
    tqdm.pandas(desc="Calculating text features")
    df['text_length'] = df['normalized_text'].progress_apply(len)
    df['word_count'] = df['normalized_text'].progress_apply(lambda x: len(x.split()))
    
    tqdm.pandas(desc="Calculating ticker counts")
    df['ticker_count'] = df['tickers'].progress_apply(len)
    
    print(f"Text preprocessing complete. Sample normalized text:")
    print(df[['TokenizedText', 'normalized_text']].head(3))
    
    return df

def validate_data(df, dataset_name):
    print(f"\nValidating {dataset_name}...")
    
    invalid_sentiments = df[~df['sentiment'].isin([-1, 0, 1])]
    if len(invalid_sentiments) > 0:
        print(f"WARNING: Found {len(invalid_sentiments)} invalid sentiment values")
        print(f"Invalid values: {invalid_sentiments['sentiment'].unique()}")
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    min_date = df['datetime'].min()
    max_date = df['datetime'].max()
    print(f"Date range: {min_date} to {max_date}")
    
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"WARNING: Found missing values:\n{missing_counts[missing_counts > 0]}")
    
    empty_tickers = df[df['tickers'].apply(lambda x: len(x) == 0)]
    if len(empty_tickers) > 0:
        print(f"WARNING: Found {len(empty_tickers)} rows with empty tickers")
    
    return df

def process_datetime(df):
    print("Processing datetime features...")
    
    tqdm.pandas(desc="Converting datetime")
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    tqdm.pandas(desc="Extracting date features")
    df['date'] = df['datetime'].progress_apply(lambda x: x.date())
    
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    
    return df

def preprocess_and_align_and_add_tickers(
    file1='data/tweets_labelled_09042020_16072020.csv',
    file2='data/Full_tokenized_DNN.csv',
    out1='data/tweets_labelled_final.csv',
    out2='data/Full_tokenized_DNN_final.csv',
    ref_file='data/sp500_constituents.csv',
    nrows=None
):

    df1 = pd.read_csv(file1, sep=';', nrows=nrows)
    df2 = pd.read_csv(file2, sep='\t', nrows=nrows)

    # Standardize column names
    df1 = df1.rename(columns={
        'id': 'ID',
        'created_at': 'datetime',
        'text': 'TokenizedText',
        'sentiment': 'sentiment'
    })
    df2 = df2.rename(columns={
        'ID': 'ID',
        'datetime': 'datetime',
        'TokenizedText': 'TokenizedText',
        'sentiment': 'sentiment'
    })

    # Convert sentiment in df1 to numeric
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    tqdm.pandas(desc="Mapping sentiment labels")
    df1['sentiment'] = df1['sentiment'].progress_apply(lambda x: sentiment_map.get(x, x))

    # Filter out tweets without valid sentiment labels
    print(f"Before sentiment filtering - df1: {len(df1)} rows, df2: {len(df2)} rows")
    
    # keep only rows with valid numeric sentiment (-1, 1)
    df1 = df1[df1['sentiment'].isin([-1, 1])]
    df2 = df2[df2['sentiment'].isin([-1, 1])]
    
    print(f"After sentiment filtering - df1: {len(df1)} rows, df2: {len(df2)} rows")
    print(f"df1 sentiment distribution: {df1['sentiment'].value_counts().sort_index().to_dict()}")
    print(f"df2 sentiment distribution: {df2['sentiment'].value_counts().sort_index().to_dict()}")

    columns = ['ID', 'datetime', 'TokenizedText', 'sentiment']
    df1 = df1[columns]
    df2 = df2[columns]

    df1 = add_ticker_column(df1, text_col='TokenizedText', ref_file=ref_file)
    df2 = add_ticker_column(df2, text_col='TokenizedText', ref_file=ref_file)

    # Remove rows with no associated tickers
    print(f"Before ticker filtering - df1: {len(df1)} rows, df2: {len(df2)} rows")
    df1 = df1[df1['tickers'].apply(lambda x: len(x) > 0)]
    df2 = df2[df2['tickers'].apply(lambda x: len(x) > 0)]
    print(f"After ticker filtering - df1: {len(df1)} rows, df2: {len(df2)} rows")

    df1 = process_datetime(df1)
    df2 = process_datetime(df2)

    df1 = validate_data(df1, "tweets_labelled dataset")
    df2 = validate_data(df2, "Full_tokenized_DNN dataset")

    df1 = preprocess_for_models(df1)
    df2 = preprocess_for_models(df2)

    train_df1, test_df1 = split_data_by_time(df1, test_size=0.2)
    train_df2, test_df2 = split_data_by_time(df2, test_size=0.2)

    # Save final versions
    df1.to_csv(out1, index=False)
    df2.to_csv(out2, index=False)
    
    train_df1.to_csv('data/tweets_labelled_final_train.csv', index=False)
    test_df1.to_csv('data/tweets_labelled_final_test.csv', index=False)
    train_df2.to_csv('data/Full_tokenized_DNN_final_train.csv', index=False)
    test_df2.to_csv('data/Full_tokenized_DNN_final_test.csv', index=False)

    return df1, df2, train_df1, test_df1, train_df2, test_df2

def load_sp500_reference(file='data/sp500_constituents.csv'):
    df = pd.read_csv(file)
    ticker_set = set(df['Symbol'].str.upper())
    name_to_ticker = {name.lower(): symbol.upper() for name, symbol in zip(df['Security'], df['Symbol'])}

    common_company_suffixes = set([
        'inc', 'inc.', 'holdings', 'limited', 'corp', 'corporation', 'plc', 'llc', 'lp', 'ltd', 'co', 'company', 'group', 'partners', 'class', 'plc.', 'llp', 'sa', 'nv', 'ag', 'ab', 'asa', 'spa', 'oyj', 'pte', 'bv', 'se', 'sas', 'sarl', 'gmbh', 'kg', 'kgaa', 'srl', 'pte.', 'co.', 's.p.a.', 's.a.', 's.a', 's.p.a', 'incorporated', 'trust', 'reit', 'nv.', 'ag.', 'ab.', 'oyj.', 'sas.', 'sarl.', 'gmbh.', 'kg.', 'kgaa.', 'srl.'
    ])
    keyword_to_ticker = {}
    for name, symbol in zip(df['Security'], df['Symbol']):
        words = name.lower().replace(',', '').replace('.', '').split()
        while words and words[-1] in common_company_suffixes:
            words.pop()
        if words:
            keyword = ' '.join(words)
            keyword_to_ticker[keyword] = symbol.upper()
    return ticker_set, name_to_ticker, keyword_to_ticker

AMBIGUOUS_TICKERS = set([
    'A', 'AA', 'AN', 'ALL', 'CAT', 'IT', 'SO', 'T', 'C', 'F', 'ARE'
])

def extract_tickers(text, ticker_set, name_to_ticker, keyword_to_ticker):
    tickers_found = set()
    # $TICKER (e.g., $AAPL)
    for match in re.findall(r'\$([A-Za-z]{1,5})', text):
        symbol = match.upper()
        if symbol in ticker_set:
            tickers_found.add(symbol)
    # TICKER as word (e.g., AAPL) - only ALL CAPS and at least 2 characters
    for match in re.findall(r'\b([A-Z]{2,5})\b', text):
        symbol = match.upper()
        if symbol in ticker_set:  # Only add if a valid ticker
            tickers_found.add(symbol)

    text_lower = text.lower()
    # Full company names
    for name, symbol in name_to_ticker.items():
        pattern = r'\b' + re.escape(name) + r"('s)?\b"
        if re.search(pattern, text_lower):
            tickers_found.add(symbol)
    for keyword, symbol in keyword_to_ticker.items():
        # Match 'apple', 'apple's' etc.
        pattern = r'\b' + re.escape(keyword) + r"('s)?\b"
        if re.search(pattern, text_lower):
            tickers_found.add(symbol)
    return sorted(tickers_found)

def add_ticker_column(df, text_col='TokenizedText', ref_file='data/sp500_constituents.csv'):
    ticker_set, name_to_ticker, keyword_to_ticker = load_sp500_reference(ref_file)
    tqdm.pandas(desc="Extracting tickers")
    df['tickers'] = df[text_col].progress_apply(
        lambda x: extract_tickers(str(x), ticker_set, name_to_ticker, keyword_to_ticker)
    )
    return df

if __name__ == "__main__":
    df1, df2, train_df1, test_df1, train_df2, test_df2 = preprocess_and_align_and_add_tickers(
        file1='data/tweets_labelled_09042020_16072020.csv',
        file2='data/Full_tokenized_DNN.csv',
        out1='data/tweets_labelled_final.csv',
        out2='data/Full_tokenized_DNN_final.csv',
        ref_file='data/sp500_constituents.csv',
        nrows=None  
    )
    print("Sample from tweets_labelled_final.csv:")
    print(df1[['TokenizedText', 'tickers']].head(10))
    print("\nSample from Full_tokenized_DNN_final.csv:")
    print(df2[['TokenizedText', 'tickers']].head(10))
