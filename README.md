# Predicting YouTube Video Views using Analytical Methods

This project aims to develop a predictive model that estimates the number of views a YouTube video will accumulate within a fixed time period after being uploaded. This can provide valuable insights to inform content creation strategy and marketing and advertising efforts.

## Data Collection
To create our minimum viable product (MVP), we utilize a Kaggle dataset that includes several months of data on daily trending YouTube videos. The dataset consists of features such as views, upload date, likes, tags, and metadata of the video creators [dataset](https://www.kaggle.com/datasets/datasnaek/youtube-new).

Subject to time constraints, we will consider augmenting this dataset with video thumbnail images and audio/visual features extracted from YouTube's public API.

## Analytical Techniques
The project will employ a combination of analytical techniques, including:

* *Regression Analysis and Machine Learning Algorithms*: To model the relationship between various features and views, we will explore a range of algorithms such as linear regression, decision trees, random forests, neural networks, and gradient boosting
* *Time Series Analysis*: To account for the temporal aspect of video popularity
* *Natural Language Processing*: To process and analyze video titles, descriptions, and tags for feature extraction (e.g., TF-IDF with n-grams featurization, or BERT)

Subject to time constraints, refinements to enhance the predictive power of the model are:

* *Computer Vision*: To extract features from video thumbnails (e.g., CNNs, autoencoders)
* *Audio/Visual Analysis*: To extract audio and visual features

## Project Impact and Overall Goal
The goal of this project is to build a predictive model that can accurately estimate the number of views a YouTube video will receive given certain features and metadata. This would assist:
* *Content creators*: To optimize their content creation and deployment strategies
* *Marketers/advertisers*: To identify high-performing videos for promotional purposes
* *YouTube*: To further understand factors that contribute most to a video's popularity
