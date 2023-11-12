import data_cleaning
import text_feature_engineering
import time_features


import pandas as pd
import datetime as dt

if __name__ == "__main__":
    # Load in the data
    US_data_file = "../data/US_youtube_trending_data.csv"
    df = pd.read_csv(US_data_file)#[0:1000]
    #df = time_features.add_new_df_cols_US(df)

    # Separate train, validation, and test datsets by time
    train_start_date = '2022-01-01'
    train_end_date = '2022-12-31'
    train_df = df[(df['publishedAt'] >= train_start_date) & (df['publishedAt'] <= train_end_date)]
    
    validation_start_date = '2023-01-01'
    validation_end_date = '2023-05-31'
    validation_df = df[(df['publishedAt'] >= validation_start_date) & (df['publishedAt'] <= validation_end_date)]

    test_start_date = '2023-06-01'
    test_df = df[(df['publishedAt'] >= test_start_date)]

    # Clean the input data
    #data_cleaner = data_cleaning.DataCleaner()
    #df.

    

    #Apply text transformations
    


    #Apply image data collection

    


    #Save train, validation, and test dataframes
    train_df.to_csv("../data/US_youtube_trending_train.csv")
    validation_df.to_csv("../data/US_youtube_trending_validation.csv")
    test_df.to_csv("../data/US_youtube_trending_test.csv")
    
      