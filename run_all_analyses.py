import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nltk
from nltk.corpus import opinion_lexicon
nltk.download('punkt')
from collections import Counter
import arabic_reshaper
from bidi.algorithm import get_display

def load_dataset(path, sort_key):
    """
    Load and sort a dataset from a CSV file.

    Args:
    path (str): Path to the CSV file.
    sort_key (str): Column name to sort by.

    Returns:
    pandas.DataFrame: Sorted DataFrame.
    """
    df = pd.read_csv(path)
    return df.sort_values(by=sort_key)

def read_words_from_file(file_path):
    """
    Read words from a text file and return them as a set.

    Args:
    file_path (str): Path to the text file.

    Returns:
    set: Set of words from the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        words = set(file.read().split('\n'))
    return words

def display_head(df, language, num_rows=2):
    """
    Display the first few rows of a DataFrame.

    Args:
    df (pandas.DataFrame): DataFrame to display.
    language (str): Name of the language for display purposes.
    num_rows (int): Number of rows to display.
    """
    print(f'{language} dataset:')
    print(df.head(num_rows))

def extract_statistics(df):
    """
    Extract general statistics from a DataFrame.

    Args:
    df (pandas.DataFrame): DataFrame to analyze.

    Returns:
    dict: Dictionary of extracted statistics.
    """
    stats = {
        'Number of Records': len(df),
        'TalkId Range': f"{df['TalkId'].min()} to {df['TalkId'].max()}",
        'Unique Speakers': df['Speaker'].nunique(),
        'Unique Keywords': df['Keywords'].nunique(),
        'Unique Content Entries': df['Content'].nunique(),
        'Unique URLs': df['URL'].nunique()
    }
    return stats

def calculate_statistics(text):
    """
    Calculate word count, sentence count, and average word count per sentence for a given text.

    Args:
    text (str): Text to analyze.

    Returns:
    tuple: Word count, sentence count, and average word count per sentence.
    """
    words = nltk.word_tokenize(text)
    sentences = nltk.sent_tokenize(text)   
    word_count = len(words)
    sentence_count = len(sentences)
    average_word_count_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
    return word_count, sentence_count, average_word_count_per_sentence

# Define common colors for English and Arabic datasets
EN_COLOR = 'red'
AR_COLOR = 'blue'

def plot_histogram_comparison(data1, data2, xlabel, ylabel, title, color1=EN_COLOR, color2=AR_COLOR, label1='English', label2='Arabic'):
    """
    Plot a histogram comparison for two datasets.

    Args:
    data1, data2 (iterable): Data to plot.
    xlabel, ylabel (str): Labels for the X and Y axes.
    title (str): Title of the plot.
    color1, color2 (str): Colors for the datasets.
    label1, label2 (str): Labels for the datasets.
    """
    plt.figure(figsize=(12, 6))
    plt.hist(data1, bins=20, color=color1, edgecolor='black', alpha=0.75, label=label1)
    plt.hist(data2, bins=20, color=color2, edgecolor='black', alpha=0.75, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_comparison(data1, data2, labels, ylabel, title):
    """
    Create side-by-side bar plots for comparison.

    Args:
    data1, data2 (pandas.DataFrame): DataFrames containing the data to plot.
    labels (list): List of labels for each subplot.
    ylabel (str): Label for the Y axis.
    title (str): Title for the entire plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11, 4)) 
    for i, (col, label) in enumerate(zip(['WordCount', 'SentenceCount', 'AvgWordCountPerSentence'], labels)):
        sns.barplot(x=['English', 'Arabic'], y=[data1[col].mean(), data2[col].mean()], ax=axes[i], palette=[EN_COLOR, AR_COLOR])
        axes[i].set_ylabel(ylabel)  
        axes[i].set_title(f'{label} Comparison') 
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def analyze_sentiment_per_talk(text, pos_words, neg_words):
    """
    Perform sentiment analysis on a given text using predefined positive and negative word sets.

    Args:
    text (str): Text to analyze.
    pos_words (set): Set of positive words.
    neg_words (set): Set of negative words.

    Returns:
    int: Sentiment score.
    """
    words = text.split()
    num_pos_words = sum(word in pos_words for word in words)
    num_neg_words = sum(word in neg_words for word in words)
    sentiment_score = num_pos_words - num_neg_words
    return sentiment_score

def analyze_sentiment_for_words(text, pos_words, neg_words):
    """
    Extract lists of positive and negative words from a given text.

    Args:
    text (str): Text to analyze.
    pos_words (set): Set of positive words.
    neg_words (set): Set of negative words.

    Returns:
    tuple: Lists of positive and negative words found in the text.
    """
    words = text.split()
    pos_word_list = [word for word in words if word in pos_words]
    neg_word_list = [word for word in words if word in neg_words]
    return pos_word_list, neg_word_list

def plot_top_words(words_column, color, title=None):
    """
    Plot the top words in a given column of words.

    Args:
    words_column (iterable): Iterable of lists of words.
    color (str): Color for the plot.
    title (str): Title of the plot.
    """
    all_words = [word for words_list in words_column for word in words_list]
    most_common_words = Counter(all_words).most_common(10)
    reshaped_words = [get_display(arabic_reshaper.reshape(word[0])) for word in most_common_words]
    plt.bar(reshaped_words, [count[1] for count in most_common_words], color=color)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.show()




def main(en_path= None,ar_path= None):
    
    # Path for Arabic lexicon
    ar_pos_word_path = 'arabic_positive_words.txt'
    ar_neg_word_path = 'arabic_negative_words.txt'

    en_df = load_dataset(en_path, 'TalkId')
    ar_df = load_dataset(ar_path, 'TalkId')

    display_head(en_df, 'English',num_rows=2)
    display_head(ar_df, 'Arabic', num_rows=2)

    en_stats = extract_statistics(en_df)
    ar_stats = extract_statistics(ar_df)

    print("English Dataset Statistics:", en_stats)
    print("Arabic Dataset Statistics:", ar_stats)

    # Apply the function to the 'Content' column of the English dataset
    en_df['WordCount'], en_df['SentenceCount'], en_df['AvgWordCountPerSentence'] = zip(*en_df['Content'].apply(calculate_statistics))
    # Apply the function to the 'Content' column of the Arabic dataset
    ar_df['WordCount'], ar_df['SentenceCount'], ar_df['AvgWordCountPerSentence'] = zip(*ar_df['Content'].apply(calculate_statistics))
    
    # Display the calculated statistics
    print('Statistics for English Dataset:')
    print(en_df[['WordCount', 'SentenceCount',  'AvgWordCountPerSentence']].describe())
    # Display the calculated statistics
    print('Statistics for Arabic Dataset:')
    print(ar_df[['WordCount', 'SentenceCount',  'AvgWordCountPerSentence']].describe())

    # Sort the 'AvgWordCountPerSentence' column in ar_df
    ar_df_sorted = ar_df.sort_values(by='AvgWordCountPerSentence')
    # Drop the last 13 rows from ar_df_sorted "outliers"
    ar_df_sorted = ar_df_sorted.iloc[:-13]

    # Drop the 13 rows and Sort based on 'AvgWordCountPerSentence' column in to keep the same size
    en_df = en_df.iloc[:-13]
    en_df_sorted = en_df.sort_values(by='AvgWordCountPerSentence')

    # Display the calculated statistics
    print('Statistics for English Dataset:')
    print(en_df_sorted[['WordCount', 'SentenceCount',  'AvgWordCountPerSentence']].describe())

    # Display the calculated statistics
    print('Statistics for Arabic Dataset:')
    print(ar_df_sorted[['WordCount', 'SentenceCount',  'AvgWordCountPerSentence']].describe())

    
    # Plot histograms for word count, sentence count, and average word count per sentence for both English and Arabic datasets
    plot_histogram_comparison(en_df_sorted['WordCount'], ar_df_sorted['WordCount'], 'Word Count', 'Frequency', 'Distribution of Word Count in Talks')
    plot_histogram_comparison(en_df_sorted['SentenceCount'], ar_df_sorted['SentenceCount'], 'Sentence Count', 'Frequency', 'Distribution of Sentence Count in Talks')
    plot_histogram_comparison(en_df_sorted['AvgWordCountPerSentence'], ar_df_sorted['AvgWordCountPerSentence'], 'Average Word Count per Sentence', 'Frequency', 'Distribution of Average Word Count per Sentence')


    # Comparison of Statistics between English and Arabic Datasets
    plot_comparison(en_df_sorted, ar_df_sorted, ['Word Count', 'Sentence Count', 'Avg Word Count per Sentence'],ylabel='Average Count', title='Comparison of Statistics between English and Arabic Datasets')

    # Download the opinion lexicon if you haven't already
    nltk.download('opinion_lexicon')

    en_positive_words = set(opinion_lexicon.positive())
    en_negative_words = set(opinion_lexicon.negative())
    # arabic positive words
    ar_positive_words = read_words_from_file(ar_pos_word_path)
    # arabic negative words
    ar_negative_words = read_words_from_file(ar_neg_word_path)

    # perform sentiment analysis
    ar_df_sorted['Sentiment'] = ar_df_sorted['Content'].apply(lambda x: analyze_sentiment_per_talk(x, ar_positive_words ,ar_negative_words))
    en_df_sorted['Sentiment'] = en_df_sorted['Content'].apply(lambda x: analyze_sentiment_per_talk(x, en_positive_words ,en_negative_words))

    # Plot the distribution of Sentiment Scores in Dataset
    plot_histogram_comparison(en_df_sorted['Sentiment'], ar_df_sorted['Sentiment'], 'Sentiment Score', 'Frequency', 'Distribution of Sentiment Scores in both Datasets')




    # Extract Positive and Negative Words
    ar_df_sorted[['Positive_Words', 'Negative_Words']] = ar_df_sorted['Content'].apply(lambda x: pd.Series(analyze_sentiment_for_words(x, ar_positive_words, ar_negative_words)))
    # Extract Positive and Negative Words
    en_df_sorted[['Positive_Words', 'Negative_Words']] = en_df_sorted['Content'].apply(lambda x: pd.Series(analyze_sentiment_for_words(x,en_positive_words ,en_negative_words)))

    # plot Positive Words
    plot_top_words(ar_df_sorted['Positive_Words'], 'green', 'Top 10 Arabic positive words')

    # plot Negative Words
    plot_top_words(ar_df_sorted['Negative_Words'], 'red', 'Top 10 Arabic negative words')

    # plot Positive Words (English)
    plot_top_words(en_df_sorted['Positive_Words'], 'green', 'Top 10 English positive words')

    # plot Negative Words (English)
    plot_top_words(en_df_sorted['Negative_Words'], 'red', 'Top 10 English negative words')


if __name__ == "__main__":

    # For train dataset
    english_path = 'eng/train/train_english.csv'
    arabic_path = 'ara/train/train_arabic.csv'
    main(en_path= english_path,ar_path= arabic_path)

    # # For test dataset
    # english_path_test = 'eng/test/test_english.csv'
    # arabic_path_test = 'ara/test/test_arabic.csv'
    # main(en_path= english_path_test,ar_path= arabic_path_test)
