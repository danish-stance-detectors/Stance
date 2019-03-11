from nltk import word_tokenize
import re


class CommentAnnotation:

    # initialises comment annotation class given json
    def __init__(self, json):

        # comment info
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
        text_ = re.sub("[^a-zA-Z]", " ", self.text) # replace with space
        # Convert all words to lower case and tokenize
        self.tokens = word_tokenize(text_.lower())


class Annotations:
    def __init__(self):
        self.annotations = []
        self.current_index = 0
        self.karma_max = 0
        self.karma_min = 0

    def add_annotation(self, annotation):
        if not annotation:
            return
        self.handle_karma(annotation)

        self.annotations.append(annotation)
        return annotation

    def handle_karma(self, annotation):
        if annotation.user_karma > self.karma_max:
            self.karma_max = annotation.user_karma
        if annotation.user_karma < self.karma_min:
            self.karma_min = annotation.user_karma

    # def handle_frequent_words(self, annotation):
    # TODO: Make histogram of frequent words per class

    def iterate(self):
        for annotation in self.annotations:
            yield annotation