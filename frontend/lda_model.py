import lyricsgenius
from nltk.corpus import stopwords
from time import sleep
import spacy
import string
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import gensim
from gensim import corpora
from gensim.corpora.dictionary import Dictionary
import nltk.data
nlp = spacy.load('en_core_web_sm')


def read_stopwords(filename):
    stopwords = []
    with open(filename) as file:
        for line in file:
            stopwords.append(line.rstrip())
    return stopwords

def clean_lyrics(lyrics):
    stop_words_custom = ['verse', 'im', 'get', '1000', '58', '60', '80', 'youre', 'youve',
                         'guitar', 'solo', 'instrumental', 'intro', 'pre', "3", "yo", "yeah", "00", "000", '04', '08',
                         '08embedshare', '40', '400', '40000', '4040', '40embedshare', '40k', '44',
                         '44654945embedshare', '45s', '4embedshare', '4l', '50', '50embedshare', '59', '5embedshare',
                         '600', '64', '65', '6embedshare', '6lack', '740', '7embedshare', '7ta', '80deca', '89',
                         '8embedshare', '8wheeler', '8wheelers', '90', '94', '95', '96',
                         '969', '970something', '979', '98', '980something', '984', '9bel', '9embedshare',
                         '9something4embedshare', "'s"]

    stop_words_jockers = list(stopwords.words('english'))
    stop_words_jockers += read_stopwords("jockers.stopwords")
    stop_words = stop_words_jockers + stop_words_custom
    punctuation_exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    cleaned_lyrics = []

    for song in lyrics:

        song = " ".join([ent.text for ent in nlp(song) if not ent.ent_type_])
        song = song.lower()
        song = song.replace(r"verse |[1|2|3]|chorus|bridge|outro","").replace("[","").replace("]","")
        song = song.lower().replace(r"instrumental|intro|guitar|solo","")
        song = song.replace("\n"," ").replace(r"[^\w\d'\s]+","").replace("efil ym fo flah","")
        song = song.replace(r"alright8embedshare|urlcopyembedcopy|no6embedshare", "")
        song = song.strip()

        updated_song = []

        for word in song.split():
            if word not in stop_words and word not in punctuation_exclude:

                if word.isalpha() and len(word)>3:
                    updated_song.append(lemma.lemmatize(word))

        cleaned_lyrics.append(updated_song)

    return cleaned_lyrics

def run_lda_model(lyrics_words, num_topics):
    dictionary = corpora.Dictionary(lyrics_words)
    corpus = [dictionary.doc2bow(text) for text in lyrics_words]
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=dictionary,
                                                num_topics=num_topics,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=20,
                                                alpha='auto',
                                                per_word_topics=True)

    topics = {}
    for i in range(num_topics):
        topics[i] = ' '.join([term for term, freq in lda_model.show_topic(i)])

    #print(list(topics.values()))
    return " ".join(list(topics.values()))

def fetch_lda_topics(spotify_list, client_access_token):
    genius = lyricsgenius.Genius(client_access_token)

    lyrics = []

    for i in range(len(spotify_list)):
        artist_name = spotify_list[i][1]
        song_name = spotify_list[i][0]
        song = genius.search_song(artist_name, song_name)
        sleep(0.5)
        if song:
            lyrics.append(song.lyrics)

    cleaned_lyrics = clean_lyrics(lyrics)

    run_lda_model(cleaned_lyrics, 20)

