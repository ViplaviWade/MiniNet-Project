import nltk
import spacy
import pickle
import pandas as pd
from pathlib import Path
from string import punctuation
from collections import Counter
from nltk.corpus import cmudict
from nltk.tokenize import word_tokenize, sent_tokenize



nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

nltk.download('punkt')
nltk.download('cmudict')

# Load the CMU pronouncing dictionary
cmud = cmudict.dict()

def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    # Tokenize text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    # Filter out punctuation from words
    words = [word for word in words if word.isalpha()]
    
    # Count total words and sentences
    total_words = len(words)
    total_sentences = len(sentences)
    
    # Estimate total syllables
    total_syllables = sum(count_syl(word, d) for word in words)
    
    # Calculate Flesch-Kincaid Grade Level
    if total_words > 0 and total_sentences > 0:
        fk_grade = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
    else:
        fk_grade = 0
    
    return fk_grade
    pass


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    if word.lower() in d:
        return len([phoneme for phoneme in d[word.lower()][0] if phoneme[-1].isdigit()])
    else:
        # Estimate syllables by counting vowel clusters if the word is not in the dictionary
        return len([char for char in word if char.lower() in 'aeiouy'])
    pass


def read_novels(path=None):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    
    if path is None:
        # path = Path.cwd().parent.parent / "novels"
        path = Path.cwd() / "p1-texts" / "novels"
    else:
        path = Path(path)
    
    novels = []
    for novel_path in path.glob("*.txt"):
        title, author, year = novel_path.stem.split('-')
        with open(novel_path, 'r', encoding='utf-8') as f:
            text = f.read()
        novels.append({
            "text": text,
            "title": title,
            "author": author,
            "year": int(year)
        })
    df = pd.DataFrame(novels)
    return df.sort_values(by="year").reset_index(drop=True)

    pass


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column, and writes 
    the resulting DataFrame to a pickle file."""
    
    def process_text(text):
        # Increase the maximum length for spaCy if the text is very long
        return nlp(text) if len(text) < nlp.max_length else nlp(text[:nlp.max_length])

    df["parsed"] = df["text"].apply(process_text)
    store_path.mkdir(parents=True, exist_ok=True)
    df.to_pickle(store_path / out_name)
    return df
    pass


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove punctuation tokens
    words = [word for word in tokens if word.isalpha()]  # Filter out punctuation and non-alphabetic tokens
    types = set(words)
    
    # Calculate Type-Token Ratio (TTR)
    return len(types) / len(words) if len(words) > 0 else 0
    pass


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results

def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    subjects = []
    for token in doc:
        if token.lemma_ == target_verb and token.dep_ == "ROOT":
            for child in token.children:
                if child.dep_ == "nsubj":
                    subjects.append(child.text)
    total_count = len(subjects)
    subject_counts = Counter(subjects)
    pmi = {subject: (count / total_count) / ((subjects.count(subject) / len(subjects)) * (total_count / len(subjects))) for subject, count in subject_counts.items()}
    return sorted(pmi.items(), key=lambda x: x[1], reverse=True)[:10]

def get_common_subjects_by_verb_pmi(df, verb="say"):
    """Prints the title of each novel and a list of the ten most common syntactic subjects of the verb ‘to say’ in the text, ordered by their Pointwise Mutual Information."""
    for i, row in df.iterrows():
        title = row["title"]
        doc = row["parsed"]
        common_subjects = subjects_by_verb_pmi(doc, verb)
        print(f"Title: {title}\nCommon Subjects of '{verb}' by PMI: {common_subjects}\n")

if __name__ == "__main__":
    pass

def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    subjects = []
    for token in doc:
        if token.lemma_ == verb and token.dep_ == "ROOT":
            for child in token.children:
                if child.dep_ == "nsubj":
                    subjects.append(child.text)
    return Counter(subjects).most_common(10)

def get_common_subjects_by_verb_count(df, verb="say"):
    """Prints the title of each novel and a list of the ten most common syntactic subjects of the verb ‘to say’ in the text, ordered by their frequency."""
    for i, row in df.iterrows():
        title = row["title"]
        doc = row["parsed"]
        common_subjects = subjects_by_verb_count(doc, verb)
        print(f"Title: {title}\nCommon Subjects of '{verb}': {common_subjects}\n")



def subject_counts(doc):
    """Extracts the most common subjects in a parsed document. Returns a list of tuples."""
    subjects = [token.text for token in doc if token.dep_ == "nsubj"]
    return Counter(subjects).most_common(10)
    pass

def get_common_subjects(df):
    """Prints the title of each novel and a list of the ten most common syntactic subjects overall in the text."""
    for i, row in df.iterrows():
        title = row["title"]
        doc = row["parsed"]
        common_subjects = subject_counts(doc)
        print(f"Title: {title}\nCommon Subjects: {common_subjects}\n")

def load_parsed_df(pickle_path=Path.cwd() / "pickles" / "parsed.pickle"):
    """Loads the DataFrame from the pickle file."""
    if not pickle_path.exists():
        print(f"Pickle file {pickle_path} does not exist.")
        return None
    
    with open(pickle_path, 'rb') as f:
        try:
            df = pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            return None
    return df



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """

    df = read_novels()  # this line will work once you have completed the read_novels function above.
    if df is not None:
        print(df.head())

    nltk.download("cmudict")

    res_ttrs = get_ttrs(df)
    print("Result of get_ttrs : ",res_ttrs)

    res_fks = get_fks(df)
    print("Result of get_fks : ",res_fks)

    res_parse = parse(df)
    print("Result of parse : ",res_parse)

    # df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(get_subjects(df))
    # df = parse(df)
    
    # Load the DataFrame from the pickle file
    loaded_df = load_parsed_df()
    
    for i, row in df.iterrows():
        print(row["title"])
        print(subject_counts(row["parsed"]))
        print("\n")

    get_common_subjects(df)
    get_common_subjects_by_verb_count(df, verb="say")
    get_common_subjects_by_verb_pmi(df, verb="say")

    # Loop to print the title of each novel and a list of the ten most common syntactic subjects of the verb 'to say' ordered by their frequency.
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "say"))
        print("\n")

    # Loop to print the title of each novel and a list of the ten most common syntactic subjects of the verb 'to say' ordered by their Pointwise Mutual Information.
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "say"))
        print("\n")

