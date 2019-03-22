from nltk import word_tokenize
import re


class CommentAnnotation:
        
    # initialises comment annotation class given json
    def __init__(self, json, test=False):

        if test:
            self.comment_id = "test"
            self.text = json
            self.tokens = word_tokenize(json.lower())
            
            # sdcq is just placeholder values
            self.sdqc_parent = "Supporting"
            self.sdqc_submission = "Supporting"
        else:
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
            text_ = re.sub("[^a-zA-ZæøåÆØÅ]", " ", self.text) # replace with space
            # Convert all words to lower case and tokenize
            self.tokens = word_tokenize(text_.lower())

class Annotations:
    def __init__(self):
        self.annotations = []
        self.current_index = 0
        # mapping from property to tuple: (min, max)
        self.min_max = {
            'karma': [0, 0],
            'txt_len': [0, 0],
            'tokens_len': [0, 0],
            'avg_word_len': [0, 0],
            'upvotes': [0, 0],
            'reply_count': [0, 0]
        }
        self.min_i = 0
        self.max_i = 1
        self.karma_max = 0
        self.karma_min = 0
        self.freq_histogram = []

    def add_annotation(self, annotation):
        if not annotation:
            return
        self.handle(self.min_max['karma'], annotation.user_karma)
        self.handle(self.min_max['txt_len'], len(annotation.text))
        word_len = len(annotation.tokens)
        if not word_len == 0:
            self.handle(self.min_max['tokens_len'], word_len)
            self.handle(self.min_max['avg_word_len'],
                        sum([len(word) for word in annotation.tokens]) / word_len)
        self.handle(self.min_max['upvotes'], annotation.upvotes)
        self.handle(self.min_max['reply_count'], annotation.reply_count)

        self.annotations.append(annotation)
        return annotation

    def handle(self, entries, property):
        if property > entries[self.max_i]:
            entries[self.max_i] = property
        if property < entries[self.min_i] or entries[self.min_i] == 0:
            entries[self.min_i] = property

    def get_min(self, key):
        return self.min_max[key][self.min_i]

    def get_max(self, key):
        return self.min_max[key][self.max_i]

    # def handle_frequent_words(self, annotation):
    # TODO: Make histogram of frequent words per class

    def make_frequent_words(self, take_count=100, use_parent_sdqc=0):
        # Most frequent words for annotation classes, string to int (word counter)

        # dictionary at idx #num is used for label #num, example: support at 0
        self.freq_histogram = [dict(),dict(),dict(),dict()]
        
        sdqc_to_int = {
            "Supporting": 0,
            "Denying": 1,
            "Querying": 2,
            "Commenting": 3
        }

        for annotation in self.annotations:
            dict_idx = sdqc_to_int[annotation.sdqc_parent] if use_parent_sdqc == 0 else sdqc_to_int[annotation.sdqc_submission]
            for token in annotation.tokens:
                current_histo = self.freq_histogram[dict_idx]
                if token in current_histo:
                    current_histo[token] = current_histo[token] + 1
                else:
                    current_histo[token] = 1
        
        for idx in range(len(self.freq_histogram)):
            keys = [(self.freq_histogram[idx][key],key) for key in self.freq_histogram[idx].keys()]
            keys.sort()
            keys.reverse()

            self.freq_histogram[idx] = keys[:take_count]


    def iterate(self):
        for annotation in self.annotations:
            yield annotation
    
    #TODO: the filter methods below can also extract all urls / quotes to get count of them as features maybe?

    # filters text of all annotations to replace reddit quotes with 'Reference'
    def filter_reddit_quotes(self):
        regex = re.compile(r"^>*\n$")
        for annotation in self.annotations:
            annotation.text = regex.sub("Reference", annotation.text)
    
    # filters text of all annotations to replace urls.
    def filter_text_urls(self):
        regex = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
        for annotation in self.annotations:
            annotation.text = regex.sub("url", annotation.text)