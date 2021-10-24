class Song:
    def __init__(self, name, artist, lyrics, topics=None):
        self.name = name
        self.artist = artist
        self.lyrics = lyrics
        self.topics = topics
        self.vader = None
        self.topic_embedding = None
        self.lyrics_embedding = None
        self.vader_embedding = None
        
    def set_topics(self, topics):
        self.topics = topics
    
    def set_topic_embedding(self, embedding):
        self.topic_embedding = embedding
        
    def set_lyrics_embedding(self, embedding):
        self.lyrics_embedding = embedding
        
    def set_vader_embedding(self, embedding):
        self.vader_embedding = embedding
        
    def set_vader(self, vader):
        self.vader = vader
        
    def get_lyrics(self):
        return self.lyrics
    
    def get_topics(self, ListForm = False):
        if ListForm:
            return self.topics.split(' ')
        return self.topics
    
    def get_vader(self, ListForm = False):
        if ListForm:
            return self.vader.split(' ')
        return self.vader
    
    def get_topic_embedding(self):
        return self.topic_embedding
    
    def get_vader_embedding(self):
        return self.vader_embedding
    
    def get_combined_embedding(self):
        return self.topic_embedding + self.vader_embedding
    
    def get_name(self):
        return self.name
    
    def get_combined(self, ListForm = False):
        if ListForm:
            return self.topics.split(' ') + self.vader.split(' ')
        return self.topics + ' ' + self.vader