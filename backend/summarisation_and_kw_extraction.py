from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
from nltk.tokenize import sent_tokenize
import nltk
import pandas as pd


def create_keyword_dict(item_id, df, n_keywords =20):
    # Filter the DataFrame to get reviews for the specified itemId
    item_reviews = df[df['itemId'] == item_id]['Review'].tolist()

    # Combine the reviews into a single text
    combined_text = ' '.join(item_reviews)

    # Tokenize the combined text into sentences
    sentences = sent_tokenize(combined_text)

    # Extract keywords from the combined text
    extracted_keywords = keywords(combined_text, words=n_keywords, lemmatize=True).split('\n')

    # Create a dictionary to store keyword occurrences
    keyword_occurrences = {keyword: [] for keyword in extracted_keywords}

    # Iterate through sentences and keywords
    for sentence in sentences:
        for keyword in extracted_keywords:
            if keyword in sentence:
                # Add the sentence to the dictionary under the keyword
                keyword_occurrences[keyword].append(sentence)

    return extracted_keywords, keyword_occurrences




def get_summary(item_id, df, proportion=0.2, max_words = 100):
    # Filter the DataFrame to get reviews for the specified itemId
    item_reviews = df[df['itemId'] == item_id]['Review'].tolist()

    # Combine the reviews into a single text
    combined_text = ' '.join(item_reviews)

    # Generate the summary
    summary_word_count = summarize(combined_text, word_count=max_words)
    summary_proportion = summarize(combined_text, ratio=proportion)
    summary = min(summary_word_count, summary_proportion, key=len)

    # Tokenize the summary into sentences
    summary_sentences = sent_tokenize(summary)

    # Remove duplicates by converting to a set and back to a list
    unique_summary_sentences = list(set(summary_sentences))

    # Reconstruct the summary with unique sentences
    unique_summary = ' '.join(unique_summary_sentences)

    return unique_summary



# ---------------- Use the below in app.py ------------------- #


# ratings = FoodReview.query.all()
# # Create a DataFrame with the desired fields
# df = pd.DataFrame([(rating.user_id, rating.item_id, rating.rating, rating.review) for rating in ratings],columns=['userId', 'itemId', 'rating', 'Review'])
# df = df.groupby(['userId', 'itemId'])['rating'].mean().reset_index()



df = pd.read_csv('ratings.csv')
# Call the function to create the keyword dictionary for item with itemId 7
item_id = 7
keyword_list, keyword_dict = create_keyword_dict(item_id, df)
print(keyword_list)
print("\n")
print(keyword_dict)
print("\n")


item_summary = get_summary(item_id, df)
print("\n")
print((item_summary))






