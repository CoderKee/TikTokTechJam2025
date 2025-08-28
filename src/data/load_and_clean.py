import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    # basic cleaning
    df.dropna(subset=['review_text'], inplace=True)
    return df

if __name__ == "__main__":
    df = load_data('../../data/google_reviews.csv')
    print(df.head())
