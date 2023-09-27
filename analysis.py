import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.chunk import ne_chunk
from collections import Counter
import matplotlib
matplotlib.use('TkAgg')  # Use the appropriate backend (TkAgg for most cases)
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim import corpora, models
from wordcloud import WordCloud

# Define functions to read the books
def read_book1():
    with open("Syrian.txt", "r", encoding="utf-8") as fileToRead:
        my_file1 = fileToRead.read()
    return my_file1

def read_book2():
    with open("MyCountry.txt", "r", encoding="utf-8") as fileToRead:
        my_file2 = fileToRead.read()
    return my_file2

# Function for sentence tokenize
def sent_fragmentation(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

# Function for word tokenize
def word_fragmentation(text):
    sentences = sent_fragmentation(text)
    words = [word_tokenize(sentence) for sentence in sentences]
    return words

# Lowercase the text
def small_letters(text):
    words = word_fragmentation(text)
    small_letter_words = [word.lower() for sentence in words for word in sentence]
    return small_letter_words

# Remove stop words
def clean_words(text):
    stop_words = set(nltk.corpus.stopwords.words("english"))
    additional_stop_words = ["like", "one", "us", "could", "still", "says", "would", "day", "even"]
    stop_words.update(additional_stop_words)
    words = small_letters(text)
    cleaned_words = [word for word in words if word not in stop_words]
    return cleaned_words

# Remove common punctuation characters (including colons)
import re

def remove_punctuation(text):
    words = clean_words(text)
    punctuation_remove = [re.sub(r"[^\w\s]", "", word) for word in words]
    punctuation_removed = [word.rstrip(":") for word in punctuation_remove]  # Remove colons
    return punctuation_removed

# Lemmatize words
def real_words(text):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    words = clean_words(text)
    real_word = [lemmatizer.lemmatize(word) for word in words]
    return real_word

# Frequent collocations
def frequent_colloc(text):
    text_object = nltk.Text(remove_punctuation(text))
    print(text_object.collocations())

# Function to calculate the 10 most frequent words
def top_10_frequent_words(text):
    words = remove_punctuation(text)  # Tokenize and convert to lowercase
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(10)
    return most_common_words


# Perform sentiment analysis
def perform_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    return sentiment_scores

# Function to extract person names using NER
def extract_person_names(text):
    person_names = []

    sentences = sent_tokenize(text)
    for sentence in sentences:
        words = word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)
        named_entities = ne_chunk(tagged_words)

        for subtree in named_entities:
            if type(subtree) == nltk.Tree and subtree.label() == 'PERSON':
                person_name = ' '.join([word for word, tag in subtree.leaves()])
                person_names.append(person_name)

    return person_names

# Function to find and print the most common person names
def find_most_common_names(text):
    person_names = extract_person_names(text)

    # Count the frequencies of person names
    name_frequencies = nltk.FreqDist(person_names)

    return name_frequencies

# Function to create a bar chart for the most common names
def create_name_frequency_chart(name_frequencies, title):
    names, frequencies = zip(*name_frequencies.most_common(10))  # Get the top 10 names and their frequencies
    plt.figure(figsize=(10, 6))
    plt.barh(names, frequencies)
    plt.xlabel("Frequency")
    plt.ylabel("Name")
    plt.title(title)
    plt.gca().invert_yaxis()  # Invert the y-axis to show the most frequent names at the top
    plt.show()

# Function to create a dispersion plot
def dispersion(text, words_to_plot):
    text_object = nltk.Text(word_tokenize(text))
    text_object.dispersion_plot(words_to_plot)

# Function to calculate keyword frequency
def keyword_frequency_analysis(text, keywords):
    words = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    keyword_counts = Counter(words)

    keyword_frequencies = {}
    for keyword in keywords:
        keyword_frequencies[keyword] = keyword_counts[keyword]

    return keyword_frequencies

# Function to perform sentiment analysis on sentences containing specific keywords
def sentiment_analysis_on_keywords(text, keywords):
    sentences = sent_fragmentation(text)
    keyword_sentences = [sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in keywords)]
    
    sentiment_scores_list = []
    for idx, sentence in enumerate(keyword_sentences):
        sentiment_scores = perform_sentiment_analysis(sentence)
        sentiment_scores_list.append((idx, sentiment_scores['compound']))

    positions, scores = zip(*sentiment_scores_list)

    plt.figure(figsize=(12, 6))

    # Color coding for sentiment categories
    colors = ['green' if score >= 0.05 else 'red' if score <= -0.05 else 'gray' for score in scores]

    # Plot sentiment scores
    plt.scatter(positions, scores, c=colors, cmap=plt.get_cmap('coolwarm'), edgecolor='k', alpha=0.7)

    # Highlight extreme sentiment scores
    for i, score in enumerate(scores):
        if abs(score) >= 0.5:
            plt.text(positions[i], scores[i], f"{score:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold', color='black')

    # Create a legend for sentiment categories
    legend_labels = ['Positive', 'Negative', 'Neutral']
    legend_colors = ['green', 'red', 'gray']
    legend_handles = [plt.Line2D([0], [0], marker='o', color=color, label=label, markersize=8, linestyle='None') for color, label in zip(legend_colors, legend_labels)]
    plt.legend(handles=legend_handles, title="Sentiment Categories", loc='upper right')

    plt.xlabel('Position in Text')
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Analysis on Sentences Containing Keywords')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Add a horizontal line at neutral sentiment (score 0)
    plt.grid(True)
    plt.show()




    
def create_dispersion_plot(text, words_to_plot):
    text_object = nltk.Text(nltk.word_tokenize(text))
    text_object.dispersion_plot(words_to_plot)

# Function to perform topic modeling on real words without punctuation and stop words
def perform_topic_modeling_on_real_words(text):
    # Preprocess text to remove punctuation and stop words
    words = remove_punctuation(text)

    # Create a dictionary and corpus
    dictionary = corpora.Dictionary([words])
    corpus = [dictionary.doc2bow(words)]

    # Create and train the LDA model
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

    # Print the topics
    topics = lda_model.print_topics(num_words=5)
    for topic in topics:
        print(topic)



# Function to create a word cloud
def create_word_cloud(text):
    # Preprocess the text (you can adjust this based on your specific needs)
    #text = ' '.join(text)  # Join all the words back into a single text string

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")  # Turn off axis labels
    plt.title("Word Cloud")
    plt.show()


def main():
    # Process the first file
    print("Processing Syrian.txt:")
    text1 = read_book1()
   
    name_frequencies1 = find_most_common_names(text1)
    create_name_frequency_chart(name_frequencies1, "Most Common Names in Syrian.txt")
    create_dispersion_plot(text1, ["war", "protest", "regime", "assad", "rebels"])
    frequent_colloc(text1)
    perform_topic_modeling_on_real_words(text1)
    

    # Perform keyword frequency analysis
    war_keywords = ["war", "protest", "assad","regime", "rebels", "syria"]
    keyword_frequencies1 = keyword_frequency_analysis(text1, war_keywords)
    print("Keyword Frequency Analysis for Syrian.txt")
    print(keyword_frequencies1)

    

    # Perform sentiment analysis on sentences containing keywords
    sentiment_analysis_on_keywords(text1, war_keywords)

    # cloud text
    create_word_cloud(text1)
   

    # Process the second file
    print("Processing MyCountry.txt:")
    text2 = read_book2()
    print("The most frequent names in MyCountry.txt")
    name_frequencies2 = find_most_common_names(text2)
    create_name_frequency_chart(name_frequencies2, "Most Common Names in MyCountry.txt")
    frequent_colloc(text2)
    perform_topic_modeling_on_real_words(text2)
    create_dispersion_plot(text2, ["war",  "regime", "assad", "rebels", "arrest"])
    

    # Perform keyword frequency analysis
    keyword_frequencies2 = keyword_frequency_analysis(text2, war_keywords)
    print("Keyword Frequency Analysis for MyCountry.txt")
    print(keyword_frequencies2)

    # Perform sentiment analysis on sentences containing keywords
    sentiment_analysis_on_keywords(text2, war_keywords)

    #cloud text
    create_word_cloud(text2)
    

if __name__ == "__main__":
    main()

