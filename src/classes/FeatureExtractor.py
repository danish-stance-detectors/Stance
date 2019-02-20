from CommentAnnotation import CommentAnnotation

# Class which can extract features from comment annotations
class FeatureExtractor:

    # extracts features from the text of the comment annotation
    def text_features(self, text):
        # number of chars
        txt_len = len(text)
        # number of words
        word_len = len(text.split())

        avg_word_len = sum([len(word) for word in text.split()]) / word_len

        #Period (.)
        period = '.' in text
        #Exclamation mark (!)
        e_mark = '!' in text
        #Question mark(?)
        q_mark = '?' in text

        #dotdotdot
        hasTripDot = '...' in text
        
        #dotdotdot count
        tripDotCount = text.count('...')
        
        #Question mark count
        q_mark_count = text.count('?')

        #Exclamation mark count
        e_mark_count = text.count('!')

        #Ratio of capital letters
        cap_count = sum(1 for c in text if c.isupper())
        cap_ratio = cap_count / len(text)
        return [int(period), 
                int(e_mark), 
                int(q_mark), 
                int(hasTripDot), 
                tripDotCount, 
                q_mark_count, 
                e_mark_count, 
                float(cap_ratio), 
                txt_len, 
                word_len,
                avg_word_len]