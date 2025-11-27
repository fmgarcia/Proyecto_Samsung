import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji

negative_words = {
    'no',
    'not',
    'none',
    'neither',
    'never',
    'nobody',
    'nothing',
    'nowhere',
    "doesn't",
    "isn't",
    "wasn't",
    "shouldn't",
    "won't",
    "can't",
    "couldn't",
    "don't",
    "haven't",
    "hasn't",
    "hadn't",
    "aren't",
    "weren't",
    "wouldn't",
    "daren't",
    "needn't",
    "didn't",
    "without",
    "against",
    "negative",
    "deny",
    "reject",
    "refuse",
    "decline",
    "unhappy",
    "sad",
    "miserable",
    "hopeless",
    "worthless",
    "useless",
    "futile",
    "disagree",
    "oppose",
    "contrary",
    "contradict",
    "disapprove",
    "dissatisfied",
    "objection",
    "unsatisfactory",
    "unpleasant",
    "regret",
    "resent",
    "lament",
    "mourn",
    "grieve",
    "bemoan",
    "despise",
    "loathe",
    "detract",
    "abhor",
    "dread",
    "fear",
    "worry",
    "anxiety",
    "sorrow",
    "gloom",
    "melancholy",
    "dismay",
    "disheartened",
    "despair",
    "dislike",
    "aversion",
    "antipathy",
    "hate",
    "disdain"
}
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))
stop_words = stop_words.difference(negative_words)
def remove_stopwords(sentence, stopwords_list):
    sentence = re.sub(r"@\w+", "", sentence)                # Menciones
    sentence = re.sub(r"http\S+|www\S+", "", sentence)      # URLs
    sentence = emoji.demojize(sentence, delimiters=(" ", " "))    # Emojis
    sentence = re.sub(r"[^\w\s#]", "", sentence)            # Puntuación (mantiene hashtags)
    sentence = re.sub(r'<\.*?>', '', sentence)               # Quitar signos de puntuación junto a palabras
    sentence = sentence.lower()                             # Minúsculas
    sentence = re.sub(r"\s+", " ", sentence).strip()        # Espacios
    tokens = nltk.word_tokenize(sentence)
    lemmatizer = WordNetLemmatizer()         
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words ]
     # Lematización
    filtered_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return ' '.join(filtered_tokens)

# Ejemplo de uso
example_text = "I do not like this product because it is useless and makes me sad."
filtered_text = remove_stopwords(example_text, stop_words)
print(filtered_text)