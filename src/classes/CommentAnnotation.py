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

    def create_feature_vector(self):
        feature_vec = list()

        feature_vec.extend(self.text_features())
        feature_vec.extend(self.user_features())
        feature_vec.extend(self.special_words_in_text())

        return (self.comment_id, self.sdqc_to_int(self.sdqc_parent), self.sdqc_to_int(self.sdqc_submission), feature_vec)
    
    def text_features(self):
        txt_len = len(self.text)
        #Period (.)
        period = '.' in self.text
        #Explamation mark (!)
        e_mark = '!' in self.text
        #Question mark(?)
        q_mark = '?' in self.text
        #Ratio of capital letters
        cap_count = sum(1 for c in self.text if c.isupper())
        cap_ratio = cap_count / len(self.text)
        return [int(period), int(e_mark), int(q_mark), float(cap_ratio), txt_len]
    
    def user_features(self):
        return [self.user_karma, int(self.user_gold_status), int(self.user_is_employee), int(self.user_has_verified_email)]

    # TODO: find special word cases
    def special_words_in_text(self):
        return []

    def sdqc_to_int(self, sdqc):
        return {
            "Supporting" : 0,
            "Denying" : 1,
            "Querying" : 2,
            "Commenting" : 3
        }[sdqc]