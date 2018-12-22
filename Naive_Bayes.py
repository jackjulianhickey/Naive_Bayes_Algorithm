import nltk
import pandas as pd
from fractions import Fraction
import math
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()
import string
import re


def split_into_full_sentences(positive_sentences, negative_sentences):
    # Read the CSV file for training data
    test = pd.read_csv('Data/train.csv', names=['ID', 'Sentiment', 'Source', 'Text'], error_bad_lines=False)

    # convert this data to a frame and then extract the relevant data to a list
    frame = pd.DataFrame(test)
    text = frame['Text'].tolist()
    sentiment = frame['Sentiment'].tolist()

    # Place these lists into a dictionary
    all_data = dict(zip(text, sentiment))

    # Split these into either negative (0) or positive sentences (1)
    for text, sentiment in all_data.items():
        if sentiment is '1':
            # Call the function to remove urls
            text = remove_urls(str(text))
            positive_sentences[str(text)] = sentiment
        else:
            # Call the function to remove urls
            text = remove_urls(str(text))
            negative_sentences[str(text)] = sentiment

    print("Removed URLS")


def split_into_words(tknzr_list, all_words, sentiment_words):
    # Create a set of stopwords from the NLTK stopwords list
    stopWords = set(stopwords.words('english'))

    # iterate through the list of tokenized words
    for words in tknzr_list:

        # lemmatize the word
        words = wordnet_lemmatizer.lemmatize(words)

        # check if the words and emoticon
        words = check_if_emoticon(words)

        # if the word hasn't been added to the dictionary, add it
        if str(words) not in sentiment_words:
            if words not in stopWords:
                sentiment_words[str(words)] = 1

        # else increase the number of occurences of that word
        else:
            if words not in stopWords:
                sentiment_words[str(words)] += 1
        if str(words) not in all_words:
            if words not in stopWords:
                all_words[str(words)] = 1
        else:
            if words not in stopWords:
                all_words[str(words)] += 1


def strip_handles(positive_sentences, negative_sentences, all_words, positive_words, negative_words):
    # Remove the handles, reduce the length and put to lowercase
    tknzr = nltk.TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)

    # get the number of positive sentences
    pos_num_sentences = len(positive_sentences)
    num_neg = 0

    # use the tokenizer on each word
    for sentences in positive_sentences:
        tknzr_list = tknzr.tokenize(sentences)

        # add word to its dictionary
        split_into_words(tknzr_list, all_words, positive_words)

    for sentences in negative_sentences:
        tknzr_list = tknzr.tokenize(sentences)
        split_into_words(tknzr_list, all_words, negative_words)

        # when the number of negative sentences processed is the same as the positive stop
        if num_neg == pos_num_sentences:
            break
        else:
            num_neg += 1

    print("stripped handles and reduced length")
    print("Removed Stopwords")
    print("Used lemmatizer")
    print("Converted emoticons to text")


def check_if_emoticon(words):
    # a list of various emoticons
    happy = [":d", ":-d", ":)", ":-)", ";)", ";-)", ":p", ":-p", ":P", ":-P", ":D", ":-D"]
    surprised = [":o", ":-o"]
    sad = [":(", ":-(", ":/", ":-/", ":'("]

    emoticons = {"happy": happy, "surprised": surprised, "sad": sad}

    # if the word matches an emoticon return that emoticons word
    for emotions, emoticons in emoticons.items():
        if words in emoticons:
            return emotions

    return words


# def split_into_words_non_tokenized(positive_sentences, negative_sentences, positive_words, negative_words, all_words):
#     stopWords = set(stopwords.words('english'))
#     lemmatizer = WordNetLemmatizer()
#     for text in positive_sentences:
#         for words in text.split():
#             # words = lemmatizer.lemmatize(words)
#             words = words.lower()
#             # words = check_if_emoticon(words)
#             if words not in positive_words:
#                 # if words not in stopWords:
#                 positive_words[words] = 1
#             else:
#                 # if words not in stopWords:
#                 positive_words[words] += 1
#
#     for text in negative_sentences:
#         for words in text.split():
#             # words = lemmatizer.lemmatize(words)
#             words = words.lower()
#             # words = check_if_emoticon(words)
#             if words not in negative_words:
#                 # if words not in stopWords:
#                 negative_words[words] = 1
#             else:
#                 # if words not in stopWords:
#                 negative_words[words] += 1
#
#     for text in positive_sentences:
#         for words in text.split():
#             # words = lemmatizer.lemmatize(words)
#             words = words.lower()
#             # words = check_if_emoticon(words)
#             if words not in all_words:
#                 # if words not in stopWords:
#                 all_words[words] = 1
#             else:
#                 # if words not in stopWords:
#                 all_words[words] += 1
#
#     for text in negative_sentences:
#         for words in text.split():
#             # words = lemmatizer.lemmatize(words)
#             words = words.lower()
#             # words = check_if_emoticon(words)
#             if words not in all_words:
#                 # if words not in stopWords:
#                 all_words[words] = 1
#             else:
#                 # if words not in stopWords:
#                 all_words[words] += 1
#
#     print("Put words to lowercase")
#     # print("Use lemmatizer")
#     # print("Converted common emoticons")


def naive_bayes_preparation(all_words, x_words, results):
    num_x_words = 0  # total words in that class
    total_unique_words = 0  # total words
    word_appearances_in_class = 0  # this is the number of times the word appears in the dictionary

    for words, keys in x_words.items():
        num_x_words += keys

    for words, keys in all_words.items():
        total_unique_words += keys

    for words, keys in all_words.items():
        for current_word, num_appearances in x_words.items():
            # if the class doesn't contain a word prepare that appropriately
            if words not in x_words:
                # function to calculate the fraction
                fraction_preparation(words, word_appearances_in_class, num_x_words, total_unique_words,
                                     results)
                word_appearances_in_class = 0
            else:
                # if the word does exist in the class classify it appropriately
                if words == current_word:
                    word_appearances_in_class = num_appearances
                    fraction_preparation(words, word_appearances_in_class, num_x_words, total_unique_words,
                                         results)
                    word_appearances_in_class = 0


def fraction_preparation(words, word_appearances_in_class, total_num_words_in_class, total_unique_words, results):
    word_appearances_in_class = word_appearances_in_class + 1
    denominator = total_num_words_in_class + total_unique_words
    ans = Fraction(word_appearances_in_class, denominator)

    # store the word and it's fraction
    results[words] = ans


def load_test_data(test_data):
    # open the test data
    test = pd.read_csv('Data/test.csv', names=['ID', 'Sentiment', 'Source', 'Text'], error_bad_lines=False)

    frame = pd.DataFrame(test)
    text = frame['Text'].tolist()
    sentiment = frame['Sentiment'].tolist()

    all_data = dict(zip(text, sentiment))

    # store it in a dictionary and put the words to lowercase
    for text, sentiment in all_data.items():
        test_data[str(text).lower()] = sentiment


def classify_data(test_data, positive_results, negative_results, probability_positive,
                  probability_negative):
    new_df = pd.DataFrame(columns=['Text', 'Sentiment'])
    sentence_id = 0
    for text, value in test_data.items():
        list_of_words = []
        # create a list of words in each sentence
        for words in text.split():
            list_of_words.append(words)

        sentence_len = len(list_of_words)
        start_int = 0

        # setup the calculation
        pos_result = math.log10((probability_positive.numerator / probability_positive.denominator))
        neg_result = math.log10((probability_negative.numerator / probability_negative.denominator))

        # while there's still words in the sentence
        while start_int < sentence_len:
            # if the word in the list is found calculate the log
            for pos_text, values in positive_results.items():
                if pos_text == list_of_words[start_int]:
                    pos_result += math.log10((values.numerator / values.denominator))

            for neg_text, values in negative_results.items():
                if neg_text == list_of_words[start_int]:
                    neg_result += math.log10((values.numerator / values.denominator))

            start_int += 1

        # calculate the sentiment
        calculate_sentiment(pos_result, neg_result, text, new_df, sentence_id)
        sentence_id += 1

    write_results(new_df)


def calculate_sentiment(pos_result, neg_result, text, new_df, sentence_id):
    # check if the sentence is positive or negative
    if pos_result > neg_result:
        new_df.loc[sentence_id] = [text, 1]
    elif neg_result > pos_result:
        new_df.loc[sentence_id] = [text, 0]


def write_results(new_df):
    new_df.to_csv("results.csv")


# calculates the accurracy of the results by comparing them to the tests sentiment
def calculate_accuracy(test_data):
    results = pd.read_csv('results.csv', names=['Text', 'Sentiment'], error_bad_lines=False)

    results_data = {}
    correct_results = 0
    incorrect_results = 0

    positive_correct_results = 0
    positive_incorrect_results = 0

    negative_correct_results = 0
    negative_incorrect_results = 0

    frame = pd.DataFrame(results)
    text = frame['Text'].tolist()
    sentiment = frame['Sentiment'].tolist()

    all_data = dict(zip(text, sentiment))

    for text, sentiment in all_data.items():
        results_data[str(text)] = sentiment

    for test_text, test_value in test_data.items():
        for result_text, result_value in results_data.items():
            if test_text in result_text:
                if int(test_value) == int(result_value):
                    correct_results += 1
                    if test_value == 1:
                        positive_correct_results += 1
                    else:
                        negative_correct_results += 1
                else:
                    incorrect_results += 1
                    if test_value == 1:
                        positive_incorrect_results += 1
                    else:
                        negative_incorrect_results += 1

    total = correct_results + incorrect_results
    total_positive = positive_correct_results + positive_incorrect_results
    total_negative = negative_correct_results + negative_incorrect_results

    percentage_correct = (correct_results / total) * 100

    percentage_positive_correct = (positive_correct_results / total_positive) * 100
    percentage_negative_correct = (negative_correct_results / total_negative) * 100

    print("total accuracy: ", percentage_correct)

    print("positive accuracy: ", percentage_positive_correct, "%")
    print("negative_accuracy: ", percentage_negative_correct, "%")


def remove_hashtags(all_words, positive_words, negative_words):
    for words in list(all_words.keys()):
        if '#' in words:
            del all_words[words]

    for words in list(positive_words.keys()):
        if '#' in words:
            del positive_words[words]

    for words in list(negative_words.keys()):
        if '#' in words:
            del negative_words[words]

    print("Removed all hashtags")


# def remove_punctuation(positive_sentences, negative_sentences):
#     table = str.maketrans({key: None for key in string.punctuation})
#
#     for sentences in positive_sentences:
#         for words in sentences.split():
#             if '@' not in words and '#' not in words:
#                 new_sentence = sentences.translate(table)
#                 positive_sentences[new_sentence] = positive_sentences.pop(sentences)
#                 break
#
#     for sentences in negative_sentences:
#         for words in sentences.split():
#             if '@' not in words and '#' not in words:
#                 new_sentence = sentences.translate(table)
#                 negative_sentences[new_sentence] = negative_sentences.pop(sentences)
#                 break
#
#     print("Removed punctuation from sentences")


def remove_non_ascii(all_words, positive_words, negative_words):
    for word in list(all_words.keys()):
        for char in word:
            if ord(char) > 127:
                del all_words[word]
                break

    for word in list(positive_words.keys()):
        for char in word:
            if ord(char) > 127:
                del positive_words[word]
                break

    for word in list(negative_words.keys()):
        for char in word:
            if ord(char) > 127:
                del negative_words[word]
                break

    print("Removed all non words without ASCII symbol")


def remove_word_punctuation(all_words, positive_words, negative_words):
    table = str.maketrans({key: None for key in string.punctuation})

    for words in list(all_words.keys()):
        new_word = words.translate(table)
        all_words[new_word] = all_words.pop(words)
        continue

    for words in list(positive_words.keys()):
        new_word = words.translate(table)
        positive_words[new_word] = positive_words.pop(words)
        continue

    for words in list(negative_words.keys()):
        new_word = words.translate(table)
        negative_words[new_word] = negative_words.pop(words)
        continue

    print("Removed punctuation from words")


def remove_urls(sentence):
    new_sentence = re.sub('(www|http)\S+', '', sentence, flags=re.MULTILINE)

    return new_sentence


def main():
    print("starting")
    positive_sentences = {}
    negative_sentences = {}
    positive_words = {}
    negative_words = {}
    all_words = {}

    split_into_full_sentences(positive_sentences, negative_sentences)

    num_positive_sentences = len(positive_sentences)
    num_negative_sentences = len(negative_sentences)

    strip_handles(positive_sentences, negative_sentences, all_words, positive_words, negative_words)

    print("pos words: ", len(positive_words))
    print("neg words: ", len(negative_words))

    remove_hashtags(all_words, positive_words, negative_words)
    remove_non_ascii(all_words, positive_words, negative_words)
    remove_word_punctuation(all_words, positive_words, negative_words)

    num_sentences = num_positive_sentences + num_negative_sentences

    probability_positive = Fraction(num_positive_sentences, num_sentences)
    probability_negative = Fraction(num_negative_sentences, num_sentences)

    positive__results = {}
    naive_bayes_preparation(all_words, positive_words, positive__results)

    negative__results = {}
    naive_bayes_preparation(all_words, negative_words, negative__results)

    test_data = {}
    load_test_data(test_data)

    total_unique_words = 0
    for words, keys in all_words.items():
        total_unique_words += keys

    total_positive_words = 0
    for words, keys in positive_words.items():
        total_positive_words += keys

    total_negative_words = 0
    for words, keys in negative_words.items():
        total_negative_words += keys

    classify_data(test_data, positive__results, negative__results, probability_positive,
                  probability_negative)

    calculate_accuracy(test_data)

    print('finished')


main()
