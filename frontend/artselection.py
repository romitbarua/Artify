from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import pickle
from ArtPiece import ArtPiece
from .Song import Song
from .Vader import *

def open_art():
    with open('Art.pickle', 'rb') as f:
        art_pieces = pickle.load(f)
    return art_pieces

def buildSong(name, topics_list, glove, artist=None):
    
    clean_topics = ' '.join([topic for topic in topics_list.split(' ') if calc_sentiment(topic)['neu'] != 1.0])
    
    # song = Song(name, artist, topics = clean_topics)

    song = Song(name , topics = clean_topics)
    
    song.set_vader(' '.join(findClosestVader(song.get_topics(False), int(len(song.get_topics(True))/3))))

    topic_embeddings = []
    for topic in song.get_topics(True):
        try:
            topic_embeddings.append(glove[topic.lower()])
        except:
            pass     
    song.set_topic_embedding(topic_embeddings)

    vader_embeddings = []
    for term in song.get_vader(True):
        try:
            vader_embeddings.append(glove[term.lower()])
        except:
            pass
    
    song.set_vader_embedding(vader_embeddings)

    return song