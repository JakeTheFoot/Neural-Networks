import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from concurrent.futures import ThreadPoolExecutor


def fetch_and_save_top_stocks(num_days, stocks_to_fetch):
    def fetch_stock_data(ticker, start_date, end_date):
        return yf.download(ticker, start=start_date, end=end_date, progress=False)

    def save_stock_data(ticker, pct_change, output_directory):
        ticker_data = pct_change[ticker].dropna().tolist()
        with open(os.path.join(output_directory, f'{ticker}_percentage_change.json'), 'w') as outfile:
            json.dump(ticker_data, outfile)

    end_date = datetime.strptime('2023-01-01', '%Y-%m-%d')
    start_date = end_date - timedelta(days=num_days)

    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_data = pd.read_html(sp500_url)
    sp500_table = sp500_data[0]
    tickers = sp500_table['Symbol'].tolist()

    with ThreadPoolExecutor() as executor:
        stock_data_futures = {ticker: executor.submit(fetch_stock_data, ticker, start_date.strftime(
            '%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) for ticker in tickers}
        stock_data_list = [future.result()
                           for future in stock_data_futures.values()]

    # Filter out empty dataframes and concatenate
    stock_data = pd.concat([data for data in stock_data_list if not data.empty and data.columns[0]
                           != 'MMM'], axis=1, keys=stock_data_futures.keys())

    open_data = stock_data['Open']
    close_data = stock_data['Close']

    pct_change = (close_data - open_data) / open_data * 100
    avg_pct_change = pct_change.mean().dropna()

    top_stocks = avg_pct_change.nlargest(stocks_to_fetch).index.tolist()

    output_directory = 'stock_data'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with ThreadPoolExecutor() as executor:
        save_futures = [executor.submit(
            save_stock_data, ticker, pct_change, output_directory) for ticker in top_stocks]
        _ = [future.result() for future in save_futures]


fetch_and_save_top_stocks(num_days=30, stocks_to_fetch=10)
