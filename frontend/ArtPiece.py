class ArtPiece:
    def __init__(self, image_loc, image_link, tags, name=None, artist=None, year=None):
        self.image_loc = image_loc
        self.image_link = image_link
        self.tags = tags
        self.name = name
        self.artist = artist
        self.year = year
        self.vader_embedding = None
        self.tag_embedding = None
        self.vader = None
        
    def get_tags(self, listForm = False, splitChar = ' '):
        if listForm:
            return self.tags.split(splitChar)
        return self.tags

    def get_vader(self, listForm = False, splitChar = ' '):
        if listForm:
            return self.vader.split(splitChar)
        return self.vader
    
    def get_combined(self, listForm = False, splitChar = ' '):
        if listForm:
            return self.tags.split(splitChar) + self.vader.split(splitChar)
        return '{} {}'.format(self.tags, self.vader)
    
    def get_combined_embedding(self):
        return self.tag_embedding + self.vader_embedding
    
    def get_vader_embedding(self):
        return self.vader_embedding
    
    def get_tag_embedding(self):
        return self.tag_embedding
    
    def set_vader_embedding(self, vader_embedding):
        self.vader_embedding = vader_embedding
    
    def set_tag_embedding(self, tag_embedding):
        self.tag_embedding = tag_embedding
        
    def set_vader(self, vader):
        self.vader = vader