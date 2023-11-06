import pandas as pd
import string
from textblob import TextBlob
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

class TextFeatureExtractor:
    
    def __init__(self, keywords):
        self.keywords = keywords

    def create_keyword_flags(self, df):
        for keyword in self.keywords:
            df[keyword] = df['title'].fillna('').str.contains(keyword, case = False).astype(int)
        return df

    def create_text_features(self, df):
        
        def get_title_length_chars(text):
            if pd.notna(text):
                return len(text)
            else:
                return 0

        def get_title_length_words(text):
            if pd.notna(text):
                return len(text.split())
            else:
                return 0

        def average_word_length(text):
            if pd.notna(text):
                words = text.split()
                if words:
                    total_length = sum(len(word) for word in words)
                    return total_length / len(words)
            return 0

        def longest_word_length(text):
            if pd.notna(text):
                words = text.split()
                if words:
                    return max(len(word) for word in words)
            return 0

        def all_words_are_uppercase(text):
            if pd.isna(text):
                return False
            words = text.split()
            return all(word.isupper() for word in words)  

        def first_word_is_uppercase(text):
            if pd.notna(text):
                words = text.split()
                if words:
                    first_word = words[0]
                    return first_word.isupper()
            return False

        def any_word_is_uppercase(text):
            if pd.notna(text):
                words = text.split()
                return any(word.isupper() for word in words)
            return False

        def proportion_words_uppercase(text):
            if pd.notna(text):
                words = text.split()
                if words:
                    uppercase_words = [word for word in words if word.isupper()]
                    return len(uppercase_words) / len(words)
            return 0.0

        def all_words_are_lowercase(text):
            if pd.isna(text):
                return False
            words = text.split()
            return all(word.islower() for word in words)  

        def identify_sentiment(text):
            if pd.isna(text):
                return -9    
            analysis = TextBlob(text)
            if analysis.sentiment.polarity > 0:
                return 1
            elif analysis.sentiment.polarity < 0:
                return 0
            else:
                return -1

        def contains_digit(text):
            if text is not None and not pd.isna(text):
                for char in text:
                    if char.isdigit():
                        return True
            return False

        def starts_with_digit(text):
            if text is not None and not pd.isna(text) and len(text) > 0:
                return text[0].isdigit()
            return False

        def identify_question_in_title(text):
            if pd.notna(text):
                return '?' in text
            return False

        def count_exclamation_marks(text):
            if pd.notna(text):
                return text.count('!')
            return 0

        def count_punctuation_marks(text):
            if pd.notna(text):
                return sum(text.count(punctuation) for punctuation in string.punctuation)
            return 0

        def count_stop_words(text):
            if pd.notna(text):
                words = text.split()
                return sum(1 for word in words if word.lower() in stop_words)
            return 0

        def proportion_stop_words(text):
            if pd.notna(text):
                words = text.split()
                if words:
                    stop_word_count = sum(1 for word in words if word.lower() in stop_words)
                    return stop_word_count / len(words)
            return 0.0

        def contains_quote(text):
            if pd.notna(text):
                return (text.count('"') == 2) or (text.count("'") == 2)
            return False

        df = self.create_keyword_flags(df)
        df['title_length_chars'] = df['title'].apply(get_title_length_chars)
        df['title_length_words'] = df['title'].apply(get_title_length_words)
        df['title_avg_word_length'] = df['title'].apply(average_word_length)
        df['title_longest_word_length'] = df['title'].apply(longest_word_length)
        df['title_all_upcase'] = df['title'].apply(all_words_are_uppercase)
        df['title_first_upcase'] = df['title'].apply(first_word_is_uppercase).astype(int)
        df['title_any_upcase'] = df['title'].apply(any_word_is_uppercase).astype(int)
        df['title_prop_upcase'] = df['title'].apply(proportion_words_uppercase)
        df['title_all_lowercase'] = df['title'].apply(all_words_are_lowercase).astype(int)
        df['title_sentiment'] = df['title'].apply(identify_sentiment)
        df['title_contains_digit'] = df['title'].apply(contains_digit).astype(int)
        df['title_starts_digit'] = df['title'].apply(starts_with_digit).astype(int)
        df['title_contains_question'] = df['title'].apply(identify_question_in_title).astype(int)
        df['title_exclamation_count'] = df['title'].apply(count_exclamation_marks)
        df['title_punctuation_count'] = df['title'].apply(count_punctuation_marks)
        df['title_stop_words_count'] = df['title'].apply(count_stop_words)
        df['title_stop_words_prop'] = df['title'].apply(proportion_stop_words)
        df['title_contains_quote'] = df['title'].apply(contains_quote).astype(int)

        return df