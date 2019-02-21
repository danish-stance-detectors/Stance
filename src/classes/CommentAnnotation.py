from nltk import word_tokenize
import re

class CommentAnnotation:

    # initialies comment annotation class given json
    def __init__(self, json):

        #comment info
        self.submission_id = json["comment"]["SubmissionID"]
        self.comment_id = json["comment"]["comment_id"]
        self.text = json["comment"]["text"]
        self.parent_id = json["comment"]["parent_id"]
        self.comment_url = json["comment"]["comment_url"]
        self.created = json["comment"]["created"]
        self.upvotes = json["comment"]["upvotes"]
        self.is_submitter = json["comment"]["is_submitter"]
        self.is_deleted = json["comment"]["is_deleted"]
        self.reply_count = json["comment"]["replies"]
        
        # user info
        self.user_id = json["comment"]["user"]["id"]
        self.user_name = json["comment"]["user"]["username"]
        self.user_created = json["comment"]["user"]["created"]
        self.user_karma = json["comment"]["user"]["karma"]
        self.user_gold_status = json["comment"]["user"]["gold_status"]
        self.user_is_employee = json["comment"]["user"]["is_employee"]
        self.user_has_verified_email = json["comment"]["user"]["has_verified_email"]

        # annotation info
        self.annotator = json["annotator"]
        self.sdqc_parent = json["comment"]["SDQC_Parent"]
        self.sdqc_submission = json["comment"]["SDQC_Submission"]
        self.certainty = json["comment"]["Certainty"]
        self.evidentiality = json["comment"]["Evidentiality"]
        self.annotated_at = json["comment"]["AnnotatedAt"]

        # Remove non-alphabetic characters and tokenize
        text_ = re.sub("[^a-zA-Z]", " ", self.text) #replace with space
        #Convert all words to lower case and tokenize
        self.tokens = word_tokenize(text_.lower())

    def create_feature_vector(self, swear_words, negation_words, word_embeddings):
        feature_vec = list()

        feature_vec.extend(self.text_features())
        feature_vec.extend(self.user_features())
        feature_vec.extend(self.special_words_in_text(swear_words, negation_words))
        feature_vec.extend(self.reddit_comment_features())
        if word_embeddings:
            feature_vec.extend(word_embeddings)

        return (self.comment_id, self.sdqc_to_int(self.sdqc_parent), self.sdqc_to_int(self.sdqc_submission), feature_vec)
    
    def text_features(self):
        # number of chars
        txt_len = len(self.text)
        # number of words
        word_len = len(self.text.split())

        avg_word_len = sum([len(word) for word in self.text.split()]) / word_len

        #Period (.)
        period = '.' in self.text
        #Exclamation mark (!)
        e_mark = '!' in self.text
        #Question mark(?)
        q_mark = '?' in self.text

        #dotdotdot
        hasTripDot = '...' in self.text
        
        #dotdotdot count
        tripDotCount = self.text.count('...')
        
        #Question mark count
        q_mark_count = self.text.count('?')

        #Exclamation mark count
        e_mark_count = self.text.count('!')

        #Ratio of capital letters
        cap_count = sum(1 for c in self.text if c.isupper())
        cap_ratio = cap_count / len(self.text)
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
    def user_features(self):
        return [self.user_karma, int(self.user_gold_status), int(self.user_is_employee), int(self.user_has_verified_email)]

    # TODO: find special word cases
    def special_words_in_text(self, swear_words, negation_words):
        split_text = self.text.split(" ")
        
        swear_count = 0
        negation_count = 0
        for word in split_text:
            w = word.strip().lower()
            if w in swear_words:
                swear_count += 1

            if w in negation_words:
                negation_count += 1
                
        return [swear_count, negation_count]

    def reddit_comment_features(self):
        return [self.upvotes, self.reply_count, int(self.is_submitter)]

    def sdqc_to_int(self, sdqc):
        return {
            "Supporting" : 0,
            "Denying" : 1,
            "Querying" : 2,
            "Commenting" : 3
        }[sdqc]