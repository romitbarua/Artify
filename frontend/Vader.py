from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from heapq import nsmallest

def calc_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)
    
def sentiment_scores(row):
    return calc_sentiment(row.Word)['compound']
    
emotion_data = pd.read_csv('emotion_terms.csv')
emotion_data['compound_score'] = emotion_data.apply(sentiment_scores, axis=1)

def findClosestVader(text, num_words):
    emotion_list = []
    compound_score = calc_sentiment(text)['compound']
    for pos in range(len(emotion_data)):
        emotion_list.append((abs(emotion_data.compound_score[pos]-compound_score), emotion_data.Word[pos]))
        
    return [x[1] for x in nsmallest(num_words, emotion_list)]