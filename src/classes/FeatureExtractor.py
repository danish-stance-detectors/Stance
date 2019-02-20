from CommentAnnotation import CommentAnnotation

# Class which can extract features from comment annotations
class FeatureExtractor:

    # Extracts features from comment annotation and extends the different kind of features to eachother.
    def create_feature_vector(self, comment,  swear_words, negation_words):
        feature_vec = list()

        feature_vec.extend(self.text_features(comment.text))
        feature_vec.extend(self.user_features(comment))
        feature_vec.extend(self.special_words_in_text(comment.text, swear_words, negation_words))
        feature_vec.extend(self.reddit_comment_features(comment))

        return (comment.comment_id, self.sdqc_to_int(comment.sdqc_parent), self.sdqc_to_int(comment.sdqc_submission), feature_vec)

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

    # TODO: Normalize user karma
    def user_features(self, comment):
        return [comment.user_karma, int(comment.user_gold_status), int(comment.user_is_employee), int(comment.user_has_verified_email)]

    # TODO: find special word cases
    def special_words_in_text(self, text, swear_words, negation_words):
        split_text = text.split(" ")
        
        swear_count = 0
        negation_count = 0
        for word in split_text:
            w = word.strip().lower()
            if w in swear_words:
                swear_count += 1

            if w in negation_words:
                negation_count += 1
                
        return [swear_count, negation_count]

    def reddit_comment_features(self, comment):
        return [comment.upvotes, comment.reply_count, int(comment.is_submitter)]

    def sdqc_to_int(self, sdqc):
        return {
            "Supporting" : 0,
            "Denying" : 1,
            "Querying" : 2,
            "Commenting" : 3
        }[sdqc]