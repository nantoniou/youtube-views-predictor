import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytz
import holidays
from scipy.stats import gmean
import datetime as dt

def change_time_to_local(df, publishedAt_col='publishedAt', timezone_str='America/Los_Angeles'):
    df[publishedAt_col] = pd.to_datetime(df[publishedAt_col])
    if df[publishedAt_col].dt.tz is None:
        # If not timezone-aware, localize to UTC
        df[publishedAt_col] = pd.to_datetime(df[publishedAt_col]).dt.tz_localize('UTC')
    else:
        # If timezone-aware, ensure it is UTC to standardize before conversion
        df[publishedAt_col] = df[publishedAt_col].dt.tz_convert('UTC')
    # Convert to the desired timezone
    local_timezone = pytz.timezone(timezone_str)
    df[publishedAt_col] = df[publishedAt_col].dt.tz_convert(local_timezone)
    return df

def add_days_on_trending(df, video_id_col='video_id', trending_date_col='trending_date'):
    # Written by GPT-4
    # Ensure the trending_date column is in datetime format
    df[trending_date_col] = pd.to_datetime(df[trending_date_col])

    # Get the earliest trending date for each video
    earliest_trending = df.groupby(video_id_col)[trending_date_col].min().reset_index()
    earliest_trending.columns = [video_id_col, 'earliest_trending_date']

    # Merge the earliest trending date back into the original dataframe
    df = df.merge(earliest_trending, on=video_id_col)

    # Calculate the number of days since the earliest trending date for each video
    df['days_on_trending'] = (df[trending_date_col] - df['earliest_trending_date']).dt.days + 1

    # Drop the 'earliest_trending_date' as it's no longer needed in the final DataFrame
    df = df.drop(columns=['earliest_trending_date'])

    return df
    
def add_days_since_published(df, video_id_col='video_id', trending_date_col='trending_date', publishedAt_col='publishedAt'):
    # Ensure the trending_date column is in datetime format
    df[trending_date_col] = pd.to_datetime(df[trending_date_col])
    df[publishedAt_col] = pd.to_datetime(df[publishedAt_col])
    df['days_since_published'] = (df[trending_date_col] - df[publishedAt_col]).dt.days
    return df

#For some reason earliest_trending_date isn't working
# def add_days_to_make_to_trending(df, earliest_trending_date_col='earliest_trending_date', publishedAt_col='publishedAt'):
#     df['days_to_make_to_trending'] = df[earliest_trending_date_col] - df[publishedAt_col]
#     return df

def add_time_of_day_variables(df, output_column_modifier="", published_at_col='publishedAt'):
    #Written by GPT-4
    # Ensure the publishedAt column is in datetime format
    df[published_at_col] = pd.to_datetime(df[published_at_col])
    # Extract the hour of day (0-23)
    df[output_column_modifier+'hour_of_day_published'] = df[published_at_col].dt.hour
    # Normalize the hour of day (0-1)
    df[output_column_modifier+'hour_published_normalized'] = df[output_column_modifier+'hour_of_day_published'] / 24
    # Categorize the time of day
    def get_time_of_day(hour):
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'
    df[output_column_modifier+'time_of_day_published'] = df[output_column_modifier+'hour_of_day_published'].apply(get_time_of_day)
    return df

def add_day_of_week(df, published_at_col='publishedAt', trending_date_col='trending_date'):
    #Written by GPT-4
    # Ensure the columns are in datetime format
    df[published_at_col] = pd.to_datetime(df[published_at_col])
    df[trending_date_col] = pd.to_datetime(df[trending_date_col])
    # Get the day name for the published day
    df['day_of_week_published'] = df[published_at_col].dt.day_name()
    # Get the day name for the trending day
    df['trending_day_of_week'] = df[trending_date_col].dt.day_name()
    return df

def add_is_weekend(df, day_of_week_published_col='day_of_week_published', trending_day_of_week_col='trending_day_of_week'):
    df["is_weekend_published"] = (df[day_of_week_published_col] == 'Saturday') | (df[day_of_week_published_col] == 'Sunday')
    df["is_weekend_trending"] = (df[trending_day_of_week_col] == 'Saturday') | (df[trending_day_of_week_col] == 'Sunday')
    return df

def add_local_time_of_day(df, published_at_col='publishedAt', timezone_str='UTC'):
    #Written by GPT-4
    # Check if the publishedAt column is already timezone-aware
    if df[published_at_col].dt.tz is None:
        # If not timezone-aware, localize to UTC
        df[published_at_col] = pd.to_datetime(df[published_at_col]).dt.tz_localize('UTC')
    else:
        # If timezone-aware, ensure it is UTC to standardize before conversion
        df[published_at_col] = df[published_at_col].dt.tz_convert('UTC')
    # Convert to the desired timezone
    local_timezone = pytz.timezone(timezone_str)
    df['local_time_of_day_published'] = df[published_at_col].dt.tz_convert(local_timezone)
    # Optionally, extract the local time from the timestamp if needed (e.g., HH:MM format)
    # df['local_time_of_day_published'] = df['local_time_of_day_published'].dt.strftime('%H:%M')
    return df

def add_US_holiday_column(df, published_at_col='publishedAt'):
    # Ensure the publishedAt column is in datetime format
    df[published_at_col] = pd.to_datetime(df[published_at_col])
    # Create an instance of the US holidays
    us_holidays = holidays.UnitedStates()
    # Determine if the publish date is a holiday in the US
    df['published_on_holiday'] = df[published_at_col].apply(lambda x: x in us_holidays)
    return df

def calculate_channel_statistics(df):
    #Partially written by GPT-4
    # Convert 'publishedAt' to datetime if it's not already
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    # Sort the DataFrame by channelID and publishedAt to ensure chronological order
    df.sort_values(by=['channelId', 'publishedAt'], ascending=True, inplace=True)
    # Initialize the columns where the statistics will be stored
    df['previous_videos_count'] = 0
    df['previous_avg_views'] = np.nan
    df['previous_avg_likes'] = np.nan
    df['previous_avg_dislikes'] = np.nan
    df['previous_geo_avg_like_dislike_ratio'] = np.nan
    df['previous_avg_days_on_trending'] = np.nan
    df['previous_avg_comment_count'] = np.nan
    df['last_video_views'] = np.nan
    df['last_video_likes'] = np.nan
    df['last_video_dislikes'] = np.nan
    df['last_video_days_on_trending'] = np.nan
    df['last_video_comment_count'] = np.nan
    # Group by channelID
    for channel_id, group in df.groupby('channelId'):
        # Track the most recent day each video was on trending
        last_day_per_video = group.groupby('video_id').tail(1)
        for index, current_row in group.iterrows():
            # Consider videos that were published before the current video's trending_date
            previous_videos = last_day_per_video[last_day_per_video['publishedAt'] < current_row['publishedAt']]
            if not previous_videos.empty:
                # Calculate statistics for previous videos
                df.at[index, 'previous_videos_count'] = previous_videos['video_id'].nunique()
                df.at[index, 'previous_avg_views'] = previous_videos['view_count'].mean()
                df.at[index, 'previous_avg_likes'] = previous_videos['likes'].mean()
                df.at[index, 'previous_avg_dislikes'] = previous_videos['dislikes'].mean()
                df.at[index, 'previous_avg_days_on_trending'] = previous_videos['days_on_trending'].mean()
                df.at[index, 'previous_avg_comment_count'] = previous_videos['comment_count'].mean()
                # Calculate the geometric mean of the like to dislike ratios
                ratios = previous_videos['likes'] / previous_videos['dislikes'].replace(0, np.nan)
                df.at[index, 'previous_geo_avg_like_dislike_ratio'] = gmean(ratios.dropna())
                # Get the statistics of the last video
                last_video = previous_videos.iloc[-1]
                df.at[index, 'last_video_views'] = last_video['view_count']
                df.at[index, 'last_video_likes'] = last_video['likes']
                df.at[index, 'last_video_dislikes'] = last_video['dislikes']
                df.at[index, 'last_video_days_on_trending'] = last_video['days_on_trending']
                df.at[index, 'last_video_comment_count'] = last_video['comment_count']
    return df

def calculate_average_days_on_trending(df):
    #Written by GPT-4
    # Ensure the publishedAt column is in datetime format
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    # Sort the DataFrame by channelID and publishedAt to ensure chronological order
    df.sort_values(by=['channelId', 'publishedAt'], ascending=True, inplace=True)
    # Initialize a dictionary to keep track of the last day on trending for each video
    last_days_on_trending = {}
    # Initialize the column for the average number of days on trending
    df['previous_avg_days_on_trending'] = pd.Series(dtype=float)
    # Iterate over the DataFrame row by row
    for index, current_row in df.iterrows():
        channel_videos = last_days_on_trending.get(current_row['channelId'], {})
        # If there are previous videos, calculate the average
        if channel_videos:
            df.at[index, 'previous_avg_days_on_trending'] = pd.Series(channel_videos.values()).mean()
        # Update the last days on trending for the current video
        channel_videos[current_row['video_id']] = current_row['last_video_days_on_trending']
        # Store the updated list back in the dictionary
        last_days_on_trending[current_row['channelId']] = channel_videos
    return df


def calculate_channel_statistics_prev_time(df, lookback_days=365):
    #Partially written by GPT-4
    # Convert 'publishedAt' to datetime if it's not already
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    # Sort the DataFrame by channelID and publishedAt to ensure chronological order
    df.sort_values(by=['channelId', 'publishedAt'], ascending=True, inplace=True)
    # Initialize the columns where the statistics will be stored
    df[f'previous_{lookback_days}_days_videos_count'] = 0
    df[f'previous_{lookback_days}_days_avg_views'] = np.nan
    df[f'previous_{lookback_days}_days_avg_likes'] = np.nan
    df[f'previous_{lookback_days}_days_avg_dislikes'] = np.nan
    df[f'previous_{lookback_days}_days_geo_avg_like_dislike_ratio'] = np.nan
    df[f'previous_{lookback_days}_days_avg_days_on_trending'] = np.nan
    df[f'previous_{lookback_days}_days_avg_comment_count'] = np.nan
    df['last_video_views'] = np.nan
    df['last_video_likes'] = np.nan
    df['last_video_dislikes'] = np.nan
    df['last_video_days_on_trending'] = np.nan
    df['last_video_comment_count'] = np.nan
    # Group by channelID
    for channel_id, group in df.groupby('channelId'):
        # Track the most recent day each video was on trending
        last_day_per_video = group.groupby('video_id').tail(1)
        for index, current_row in group.iterrows():
            # Consider videos that were published before the current video's trending_date
            previous_videos_lt_cur = last_day_per_video['publishedAt'] < current_row['publishedAt']
            previous_videos_gt_lb = last_day_per_video['publishedAt'] >= (current_row['publishedAt'] - dt.timedelta(days=lookback_days))
            previous_videos = last_day_per_video[previous_videos_lt_cur & previous_videos_gt_lb]
            #print(f"len(previous_videos): {len(previous_videos)}")
            if not previous_videos.empty:
                # Calculate statistics for previous videos
                df.at[index, f'previous_{lookback_days}_days_videos_count'] = previous_videos['video_id'].nunique()
                df.at[index, f'previous_{lookback_days}_days_avg_views'] = previous_videos['view_count'].mean()
                df.at[index, f'previous_{lookback_days}_days_avg_likes'] = previous_videos['likes'].mean()
                df.at[index, f'previous_{lookback_days}_days_avg_dislikes'] = previous_videos['dislikes'].mean()
                df.at[index, f'previous_{lookback_days}_days_avg_days_on_trending'] = previous_videos['days_on_trending'].mean()
                df.at[index, f'previous_{lookback_days}_days_avg_comment_count'] = previous_videos['comment_count'].mean()
                # Calculate the geometric mean of the like to dislike ratios
                ratios = previous_videos['likes'] / previous_videos['dislikes'].replace(0, np.nan)
                df.at[index, f'previous_{lookback_days}_days_geo_avg_like_dislike_ratio'] = gmean(ratios.dropna())
                # Get the statistics of the last video
                last_video = previous_videos.iloc[-1]
                df.at[index, 'last_video_views'] = last_video['view_count']
                df.at[index, 'last_video_likes'] = last_video['likes']
                df.at[index, 'last_video_dislikes'] = last_video['dislikes']
                df.at[index, 'last_video_days_on_trending'] = last_video['days_on_trending']
                df.at[index, 'last_video_comment_count'] = last_video['comment_count']
    return df


def calculate_channel_statistics_prev_time_log(df, lookback_days=365):
    #Partially written by GPT-4
    # Convert 'publishedAt' to datetime if it's not already
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    # Sort the DataFrame by channelID and publishedAt to ensure chronological order
    df.sort_values(by=['channelId', 'publishedAt'], ascending=True, inplace=True)
    # Initialize the columns where the statistics will be stored
    df[f'previous_{lookback_days}_days_videos_count'] = 0
    df[f'previous_{lookback_days}_days_avg_views'] = np.nan
    df[f'previous_{lookback_days}_days_avg_likes'] = np.nan
    df[f'previous_{lookback_days}_days_avg_dislikes'] = np.nan
    df[f'previous_{lookback_days}_days_geo_avg_like_dislike_ratio'] = np.nan
    df[f'previous_{lookback_days}_days_avg_days_on_trending'] = np.nan
    df[f'previous_{lookback_days}_days_avg_comment_count'] = np.nan
    
    #Add a few aggregate log columns
    df[f'previous_{lookback_days}_days_log_videos_count'] = 0
    df[f'previous_{lookback_days}_days_log_avg_views'] = np.nan
    df[f'previous_{lookback_days}_days_log_avg_likes'] = np.nan
    df[f'previous_{lookback_days}_days_log_avg_dislikes'] = np.nan
    df[f'previous_{lookback_days}_days_log_avg_days_on_trending'] = np.nan
    df[f'previous_{lookback_days}_days_log_avg_comment_count'] = np.nan
    
    df['last_video_views'] = np.nan
    df['last_video_likes'] = np.nan
    df['last_video_dislikes'] = np.nan
    df['last_video_days_on_trending'] = np.nan
    df['last_video_comment_count'] = np.nan
    
    #Add a few previous video log columns
    df['last_video_log_views'] = np.nan
    df['last_video_log_likes'] = np.nan
    df['last_video_log_dislikes'] = np.nan
    df['last_video_log_days_on_trending'] = np.nan
    df['last_video_log_comment_count'] = np.nan
    
    # Group by channelID
    for channel_id, group in df.groupby('channelId'):
        # Track the most recent day each video was on trending
        last_day_per_video = group.groupby('video_id').tail(1)
        for index, current_row in group.iterrows():
            # Consider videos that were published before the current video's trending_date
            previous_videos_lt_cur = last_day_per_video['publishedAt'] < current_row['publishedAt']
            previous_videos_gt_lb = last_day_per_video['publishedAt'] >= (current_row['publishedAt'] - dt.timedelta(days=lookback_days))
            previous_videos = last_day_per_video[previous_videos_lt_cur & previous_videos_gt_lb]
            #print(f"len(previous_videos): {len(previous_videos)}")
            if not previous_videos.empty:
                # Calculate statistics for previous videos
                df.at[index, f'previous_{lookback_days}_days_videos_count'] = previous_videos['video_id'].nunique()
                df.at[index, f'previous_{lookback_days}_days_avg_views'] = previous_videos['view_count'].mean()
                df.at[index, f'previous_{lookback_days}_days_avg_likes'] = previous_videos['likes'].mean()
                df.at[index, f'previous_{lookback_days}_days_avg_dislikes'] = previous_videos['dislikes'].mean()
                df.at[index, f'previous_{lookback_days}_days_avg_days_on_trending'] = previous_videos['days_on_trending'].mean()
                df.at[index, f'previous_{lookback_days}_days_avg_comment_count'] = previous_videos['comment_count'].mean()
                # Calculate the geometric mean of the like to dislike ratios
                ratios = previous_videos['likes'] / previous_videos['dislikes'].replace(0, np.nan)
                df.at[index, f'previous_{lookback_days}_days_geo_avg_like_dislike_ratio'] = gmean(ratios.dropna())

                #Do the same with the log features
                df.at[index, f'previous_{lookback_days}_days_log_videos_count'] = np.log(previous_videos['video_id'].nunique() + 1)
                df.at[index, f'previous_{lookback_days}_days_log_avg_views'] = np.log(previous_videos['view_count'] + 1).mean()
                df.at[index, f'previous_{lookback_days}_days_log_avg_likes'] = np.log(previous_videos['likes'] + 1).mean()
                df.at[index, f'previous_{lookback_days}_days_log_avg_dislikes'] = np.log(previous_videos['dislikes'] + 1).mean()
                df.at[index, f'previous_{lookback_days}_days_log_avg_days_on_trending'] = np.log(previous_videos['days_on_trending'] + 1).mean()
                df.at[index, f'previous_{lookback_days}_days_log_avg_comment_count'] = np.log(previous_videos['comment_count'] + 1).mean()
                
                # Get the statistics of the last video
                last_video = previous_videos.iloc[-1]
                df.at[index, 'last_video_views'] = last_video['view_count']
                df.at[index, 'last_video_likes'] = last_video['likes']
                df.at[index, 'last_video_dislikes'] = last_video['dislikes']
                df.at[index, 'last_video_days_on_trending'] = last_video['days_on_trending']
                df.at[index, 'last_video_comment_count'] = last_video['comment_count']
                
                df.at[index, 'last_video_log_views'] = np.log(last_video['view_count'])
                df.at[index, 'last_video_log_likes'] = np.log(last_video['likes'])
                df.at[index, 'last_video_log_dislikes'] = np.log(last_video['dislikes'])
                df.at[index, 'last_video_log_days_on_trending'] = np.log(last_video['days_on_trending'])
                df.at[index, 'last_video_log_comment_count'] = np.log(last_video['comment_count'])
                
    return df


##### EXAMPLE USAGE

data_folder = '../data/'
files = [
    'US_youtube_trending_data_cleaned.csv',
    'CA_youtube_trending_data_cleaned.csv',
    'GB_youtube_trending_data_cleaned.csv',
]
country = ['US', 'CA', 'GB']
country_to_timezone = {
    'US':'America/New_York',
    'CA':'America/New_York',
    'GB':'Europe/London',
}

stem = "plus_time_info_"

def add_new_df_cols(files, data_folder, countries):
    """
    Add new columns to all dataframes
    """
    dfs = []
    for file, country in zip(files, countries):
        filename = data_folder + file
        df = pd.read_csv(filename)
        df = add_days_on_trending(df)
        df = add_days_since_published(df)
        df = add_time_of_day_variables(df)
        df = add_day_of_week(df)
        df = add_is_weekend(df)
        df = add_local_time_of_day(df, timezone_str=country_to_timezone[country])
        df = add_time_of_day_variables(df, output_column_modifier="local_", published_at_col='local_time_of_day_published')
        df = add_US_holiday_column(df)
        df = calculate_channel_statistics_prev_time(df, lookback_days=365)
        #df = calculate_channel_statistics(df)
        df = calculate_average_days_on_trending(df)
        new_filename = data_folder + stem + file
        print(f"Saving {new_filename}")
        df.to_csv(new_filename, index=False)
        dfs.append(df)
    return dfs

def add_new_df_cols_US(df):
    """
    Add new columns to the US dataframes
    """
    df = change_time_to_local(df, timezone_str='America/Los_Angeles')
    df = add_days_on_trending(df)
    df = add_days_since_published(df)
    df = add_time_of_day_variables(df)
    df = add_day_of_week(df)
    df = add_is_weekend(df)
    #df = add_local_time_of_day(df, timezone_str='America/Los_Angeles')
    #df = add_time_of_day_variables(df, output_column_modifier="local_", published_at_col='local_time_of_day_published')
    df = add_US_holiday_column(df)
    df = calculate_channel_statistics_prev_time_log(df, lookback_days=365)
    #df = calculate_channel_statistics_prev_time(df, lookback_days=365)
    #df = calculate_channel_statistics(df)
    df = calculate_average_days_on_trending(df)
    return df
