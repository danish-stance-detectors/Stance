from nltk import word_tokenize
import re

url_tag = 'URLURLURL'
quote_tag = 'REFREFREF'

class RedditAnnotation:
        
    # initialises comment annotation class given json
    def __init__(self, json, is_source=False, test=False):
        self.is_source = is_source

        if test:
            self.comment_id = "test"
            self.text = json
            self.tokens = word_tokenize(json.lower())
            
            # sdcq is just placeholder values
            self.sdqc_parent = "Supporting"
            self.sdqc_submission = "Supporting"
            return

        comment_json = json["comment"] if not is_source else json
        self.text = comment_json["text"]
        self.text = self.filter_reddit_quotes(self.text)
        self.text = self.filter_text_urls(self.text)
        if is_source:
            self.comment_id = comment_json["submission_id"]
            self.title = json["title"]
            self.num_comments = json["num_comments"]
            self.url = json["url"]
            self.text_url = json["text_url"]
            self.is_video = json["is_video"]
            # self.subreddit = json["subreddit"] # irrelevant
            # self.comments = json["comments"] # irrelevant
            self.reply_count = comment_json["num_comments"]
            self.is_submitter = True
            self.is_rumour = json["IsRumour"]
            self.is_irrelevant = json["IsIrrelevant"]
            self.truth_status = json["TruthStatus"]
            self.rumour = json["RumourDescription"]
            sdqc_source = json["SourceSDQC"]
            sdqc = "Commenting" if sdqc_source == "Underspecified" else sdqc_source
            self.sdqc_parent = sdqc
            self.sdqc_submission = sdqc
            self.tokens = self.tokenize(self.title)
        else:
            # comment specific info
            self.comment_id = comment_json["comment_id"]
            self.parent_id = comment_json["parent_id"]
            self.comment_url = comment_json["comment_url"]
            self.is_submitter = comment_json["is_submitter"]
            self.is_deleted = comment_json["is_deleted"]
            self.reply_count = comment_json["replies"]
            self.tokens = self.tokenize(comment_json["text"])

            # annotation info
            self.annotator = json["annotator"]
            self.sdqc_parent = comment_json["SDQC_Parent"]
            self.sdqc_submission = comment_json["SDQC_Submission"]
            self.certainty = comment_json["Certainty"]
            self.evidentiality = comment_json["Evidentiality"]
            self.annotated_at = comment_json["AnnotatedAt"]

        # general info
        self.submission_id = comment_json["submission_id"]
        self.created = comment_json["created"]
        self.upvotes = comment_json["upvotes"]

        # user info
        self.user_id = comment_json["user"]["id"]
        self.user_name = comment_json["user"]["username"]
        self.user_created = comment_json["user"]["created"]
        self.user_karma = comment_json["user"]["karma"]
        self.user_gold_status = comment_json["user"]["gold_status"]
        self.user_is_employee = comment_json["user"]["is_employee"]
        self.user_has_verified_email = comment_json["user"]["has_verified_email"]

    def tokenize(self, text):
        # Remove non-alphabetic characters and tokenize
        text_ = re.sub("[^a-zA-ZæøåÆØÅ0-9]", " ", text)  # replace with space
        # Convert all words to lower case and tokenize
        return word_tokenize(text_.lower())

    # TODO: the filter methods below can also extract all urls / quotes to get count of them as features maybe?
    # TODO: Test further
    def filter_reddit_quotes(self, text):
        """filters text of all annotations to replace reddit quotes with 'REFREFREF'"""
        # TODO: Doesn't seem to replace quotes? Also, do we want it? Also quotes are sometimes " "
        return re.sub(r"^>*\n$", quote_tag, text)

    def filter_text_urls(self, text):
        """filters text of all annotations to replace 'URLURLURL'"""
        # TODO: Doesn't catch www.
        regex = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
        return regex.sub(url_tag, text)


class RedditSubmission:
    def __init__(self, source):
        self.source = source
        self.branches = []

    def add_annotation_branch(self, annotation_branch):
        """Add a branch as a list of annotations to this submission"""
        self.branches.append(annotation_branch)


class RedditDataset:
    def __init__(self):
        self.annotations = [] # purely for testing purposes
        self.submissions = []
        self.last_submission = lambda: len(self.submissions) - 1
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
        # dictionary at idx #num is used for label #num, example: support at 0
        self.freq_histogram = [dict(), dict(), dict(), dict()]
        self.bow = set()
        self.sdqc_to_int = {
            "Supporting": 0,
            "Denying": 1,
            "Querying": 2,
            "Commenting": 3
        }

    def add_annotation(self, annotation):
        """Add to self.annotations. Should only be uses for testing purposes"""
        self.annotations.append(self.analyse_annotation(annotation))

    def add_reddit_submission(self, source):
        self.submissions.append(RedditSubmission(RedditAnnotation(source, is_source=True)))

    def add_submission_branch(self, branch, sub_sample=False):
        annotation_branch = []
        if sub_sample:
            comments = 0
            for annotation in branch:
                sdqc = annotation["comment"]["SDQC_Submission"]
                if self.sdqc_to_int[sdqc] == 3:
                    comments += 1
            if comments == len(branch):
                print("Filtered", comments)
                return
        for annotation in branch: # TODO: Skip existing annotations
            annotation = RedditAnnotation(annotation)
            self.analyse_annotation(annotation)
            annotation_branch.append(annotation)
        self.submissions[self.last_submission()].add_annotation_branch(annotation_branch)

    def print_status_report(self):
        histogram = {
            0: 0,
            1: 0,
            2: 0,
            3: 0
        }
        n = 0
        for annotation in self.iterate_annotations():
            histogram[self.sdqc_to_int[annotation.sdqc_submission]] += 1
            n += 1
        for label, count in histogram.items():
            print('{0}: {1} ({2})'.format(label, count, float(count)/float(n)))


    def analyse_annotation(self, annotation):
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
        self.handle_frequent_words(annotation)
        self.handle_bow(annotation.tokens)

        return annotation

    def handle(self, entries, prop):
        if prop > entries[self.max_i]:
            entries[self.max_i] = prop
        if prop < entries[self.min_i] or entries[self.min_i] == 0:
            entries[self.min_i] = prop

    def get_min(self, key):
        return self.min_max[key][self.min_i]

    def get_max(self, key):
        return self.min_max[key][self.max_i]

    # TODO: Make histogram of frequent words per class
    def handle_frequent_words(self, annotation, use_parent_sdqc=False):
        # Most frequent words for annotation classes, string to int (word counter)
        dict_idx = self.sdqc_to_int[annotation.sdqc_parent] \
            if use_parent_sdqc else self.sdqc_to_int[annotation.sdqc_submission]
        for token in annotation.tokens:
            current_histo = self.freq_histogram[dict_idx]
            if token in current_histo:
                current_histo[token] = current_histo[token] + 1
            else:
                current_histo[token] = 1

    # TODO: Refactor to RedditDataset to make BOW dynamically?
    def handle_bow(self, annotation_tokens):
        for t in annotation_tokens:
            self.bow.add(t)

    def get_frequent_words(self, take_count=100):
        histogram = {}
        for idx in range(len(self.freq_histogram)):
            keys = [(self.freq_histogram[idx][key], key) for key in self.freq_histogram[idx].keys()]
            keys.sort()
            keys.reverse()

            histogram[idx] = keys[:take_count]
        return histogram

    def iterate_annotations(self):
        for submission in self.submissions:
            for branch in submission.branches:
                for annotation in branch:
                    yield annotation

    def iterate_branches(self, with_source=True):
        for submission in self.submissions:
            for branch in submission.branches:
                if with_source:
                    yield submission.source, branch
                else:
                    yield branch

    def iterate_submissions(self):
        for submission in self.submissions:
            yield submission

    def size(self):
        n = 0
        for annotation in self.iterate_annotations():
            n += 1
        return n
