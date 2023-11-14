import data_cleaning
import text_feature_engineering
import time_features

import pandas as pd
import datetime as dt

if __name__ == "__main__":
    # Load in the data
    US_data_file = "../data/US_youtube_trending_data.csv"
    US_category_file = "../data/US_category_id.json"
    thumbnail_info = "../data/all_thumbnails_info.csv"

    df = pd.read_csv(US_data_file)#[0:1000]
    us_cat = pd.read_json(US_category_file)
    thumbnail_df = pd.read_csv(thumbnail_info)
    #There is a column we don't want to include, drop it
    thumbnail_df = thumbnail_df.drop(thumbnail_df.columns[0], axis=1)
    #it appears that some duplicate images got in there somehow, remove them
    thumbnail_df = thumbnail_df.drop_duplicates(subset="video_id", keep='first')

    #Apply time features to the entire dataset
    df = time_features.add_new_df_cols_US(df)
    
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
    # trending_data, category_id
    data_cleaner = data_cleaning.DataCleaner(train_df, us_cat)
    data_cleaner.clean_transform()
    train_df = data_cleaner.tranform_views(method="StandardScaler", fit=True, df=train_df)
    #Do the same for the other columns
    validation_df = data_cleaner.tranform_views(method="StandardScaler", fit=False, df=validation_df)
    test_df = data_cleaner.tranform_views(method="StandardScaler", fit=False, df=test_df)
    
    #Apply text transformations
    text_feature_extractor = text_feature_engineering.TextFeatureExtractor(keywords=[])
    
    train_df = text_feature_extractor.create_text_features(train_df)
    validation_df = text_feature_extractor.create_text_features(validation_df)
    test_df = text_feature_extractor.create_text_features(test_df)
    
    #Apply image data collection
    #Merge thumbnail_df into train_df 
    train_df = train_df.merge(thumbnail_df, how='left', on='video_id')
    validation_df = validation_df.merge(thumbnail_df, how='left', on='video_id')
    test_df = test_df.merge(thumbnail_df, how='left', on='video_id')
    
    #Save train, validation, and test dataframes
    # lineterminator='\r\n' makes it able to be read without the lineterminator='\n' in pd.read_csv()
    train_df.to_csv("../data/US_youtube_trending_train.csv", index=False, lineterminator='\r\n')
    validation_df.to_csv("../data/US_youtube_trending_validation.csv", index=False, lineterminator='\r\n')
    test_df.to_csv("../data/US_youtube_trending_test.csv", index=False, lineterminator='\r\n')



    
    
      