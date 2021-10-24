from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import pickle
from .Song import Song
from .ArtPiece import ArtPiece
from .Vader import *

def open_art(glove):
    art_links = pd.read_csv('art_links.csv')
    art_pieces = []
    for pos in range(len(art_links)):
        art_pieces.append(ArtPiece('imgLoc', art_links.Link[pos], art_links.Tags[pos]))
        
    for art in art_pieces:
        art.set_vader(' '.join(findClosestVader(art.get_tags(False), int(len(art.get_tags(True))/3))))
        tag_embeddings = []
        vader_embeddings = []
        for term in art.get_tags(True):
            try:
                tag_embeddings.append(glove[term.lower()])
            except:
                pass
        art.set_tag_embedding(tag_embeddings)
        for term in art.get_vader(True):
            try:
                vader_embeddings.append(glove[term.lower()])
            except:
                pass
        art.set_vader_embedding(vader_embeddings)

    return art_pieces
    
    # with open('/Users/gautham/Documents/Documents - gBookPro/Berkeley MIMS/CalHacks/prototype/Artify/Art.pickle', 'rb') as f:
    #     art_pieces = pickle.load(f)
    # return art_pieces

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


def compareSongArt(song, art, glove):
    
    cosine_score = 0
    song_emb = song.get_combined_embedding()
    art_emb = art.get_combined_embedding()
    for i in range(len(song_emb)):
        for j in range(len(art_emb)):
            #print(glove.cosine_similarities(song_emb, [art_emb]))
            cosine_score += glove.cosine_similarities(song_emb[i], [art_emb[j]])
            
    cosine_score = cosine_score/(len(song.get_combined_embedding()) * len(art.get_combined_embedding()))
    return cosine_score

def getBestArt(song, art_pieces, glove):
    
    max_cosine = 0 
    best_art = None

    for art in art_pieces:
        cosine_score = compareSongArt(song, art, glove)
        if cosine_score > max_cosine:
            max_cosine = cosine_score
            best_art = art
    
    return best_art