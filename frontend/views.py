from django.shortcuts import render
from django.http import HttpResponse
from .lda_model import fetch_lda_topics
from .artselection import *
from gensim.models import Word2Vec, KeyedVectors
from .neural import get_neural_style
from django.conf import settings

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
# import pandas as pd
import json

CLIENT_ID  = '6da6d0375bcc42b5aee00dc5ef978d8d'
CLIENT_SECRET  = '09378437ba0445ef8411bed1c3ec76ed'
USERNAME = 'qlwcqn7y0j7k5o94h5qr54nwx'
URI = 'http://localhost:8080/'
scope = 'user-read-currently-playing'
client_access_token = "vqYbB498w-UVWrx_Kltx2san_Y0p4aYekxT2NmQNWVujjwk70bvAg0LcZMmC9kNw"

glove = KeyedVectors.load_word2vec_format(settings.BASE_DIR_PATH + "data/glove.6B.100d.100K.w2v.txt", binary=False)
art_list = open_art(glove)

def index(request):
    # return HttpResponse("Hello, world. You're at the polls index.")
    return render(request, 'index.html')


def login(request):

    auth = SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET,redirect_uri=URI,  username=USERNAME)

    client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET) 
    #sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET,redirect_uri=URI,  username=USERNAME, scope=scope))

    # print(sp.playlist('5EUmcvQrnMPV0RjkdvgdgO'))
    print(sp.current_user_playing_track()['item']['name'])

    current_song = sp.current_user_playing_track()['item']['name']

    track_id = sp.current_user_playing_track()['item']['id']
    song_artist = sp.track(track_id)['artists'][0]['name']

    recs = sp.recommendations(seed_tracks=[track_id])

    # print(recs)

    spotify_list = []

    for track in recs['tracks'][:5]:
        # print('{}: {}'.format(track['name'], track['id']))

        id = track['id']
        artist_name = sp.track(id)['artists'][0]['name']
        spotify_list.append((artist_name, track['name']))

    # track_name = track['name']

    lda_topics = fetch_lda_topics(spotify_list, client_access_token, 20)

    song = buildSong(current_song, topics_list = lda_topics, glove = glove)

    best_art = getBestArt(song, art_list, glove)

    print(best_art.image_link)

    get_neural_style(best_art.image_link).save("./static/images/Sample_final.png")

    final_score = {
        # 'final_image' : final_image,
        'song_name' : current_song,
        'song_artist' : song_artist,
        'style_image' : best_art.image_link,
        'best_art' : best_art.get_tags(True)
    }

    return render(request, 'result.html', {'items' : json.loads(json.dumps(final_score))})
