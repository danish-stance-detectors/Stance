from nltk import word_tokenize
import re, copy
import word_embeddings
from sklearn.model_selection import train_test_split
from classes.afinn_sentiment import get_afinn_sentiment

url_tag = 'urlurlurl'
regex_url = re.compile(
    r"([(\[]?(https?://)|(https?://www.)|(www.))(?:[a-zæøåA-ZÆØÅ]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
quote_tag = 'refrefref'
regex_quote = re.compile(r">(.+?)\n")


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

        self.sim_to_src = 0
        self.sim_to_prev = 0
        self.sim_to_branch = 0
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
            self.tokens = self.tokenize(self.text)

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
        return word_tokenize(text_.lower(), language='danish')

    def filter_reddit_quotes(self, text):
        """filters text of all annotations to replace reddit quotes with 'refrefref'"""
        return regex_quote.sub(quote_tag, text)

    def filter_text_urls(self, text):
        """filters text of all annotations to replace 'URLURLURL'"""
        return regex_url.sub(url_tag, text)

    def alter_id_and_text(self, threshold=0.8, words_to_replace=2, early_stop=True):
        self.comment_id = self.comment_id + '_'
        # idx = randint(0, len(self.tokens)-1)
        best_candidates = []
        for idx in range(len(self.tokens)):
            token_old = self.tokens[idx]
            if token_old == quote_tag.lower() or token_old == url_tag.lower():
                continue
            sim_word, sim = word_embeddings.most_similar_word(token_old)[0]
            if sim_word == token_old:
                continue
            if sim > threshold:
                best_candidates.append((idx, sim_word, sim))
            if early_stop and len(best_candidates) == words_to_replace:
                break
        if not early_stop:
            best_candidates.sort(key=lambda tup: tup[2])
        for idx, token_new, _ in best_candidates[:words_to_replace]:
            self.tokens[idx] = token_new


class RedditSubmission:
    def __init__(self, source):
        self.source = source
        self.branches = []

    def add_annotation_branch(self, annotation_branch):
        """Add a branch as a list of annotations to this submission"""
        self.branches.append(annotation_branch)


def compute_similarity(annotation, previous, source, branch_tokens, is_source=False):
    # TODO: exclude itself???
    annotation.sim_to_branch = word_embeddings.cosine_similarity(annotation.tokens, branch_tokens)
    if not is_source:
        annotation.sim_to_src = word_embeddings.cosine_similarity(annotation.tokens, source.tokens)
        annotation.sim_to_prev = word_embeddings.cosine_similarity(annotation.tokens, previous.tokens)


class RedditDataset:
    def __init__(self):
        self.annotations = {}
        self.anno_to_branch_tokens = {}
        self.anno_to_prev = {}
        self.anno_to_source = {}
        self.submissions = []
        self.last_submission = lambda: len(self.submissions) - 1
        # mapping from property to tuple: (min, max)
        self.min_max = {
            'karma': [0, 0],
            'txt_len': [0, 0],
            'tokens_len': [0, 0],
            'avg_word_len': [0, 0],
            'upvotes': [0, 0],
            'reply_count': [0, 0],
            'afinn_score': [0,0],
            'url_count': [0, 0],
            'quote_count': [0, 0],
            'cap_sequence_max_len': [0, 0],
            'tripDotCount': [0, 0],
            'q_mark_count': [0, 0],
            'e_mark_count': [0, 0],
            'cap_count': [0, 0]
        }
        self.min_i = 0
        self.max_i = 1
        self.karma_max = 0
        self.karma_min = 0
        # dictionary at idx #num is used for label #num, example: support at 0
        self.freq_histogram = [dict(), dict(), dict(), dict()]
        self.bow = set()
        self.freq_tri_gram = [dict(), dict(), dict(), dict()]
        self.sdqc_to_int = {
            "Supporting": 0,
            "Denying": 1,
            "Querying": 2,
            "Commenting": 3
        }

    def add_annotation(self, annotation):
        """Add to self.annotations. Should only be uses for testing purposes"""
        annotation = self.analyse_annotation(RedditAnnotation(annotation))
        if annotation.comment_id not in self.annotations:
            self.annotations[annotation.comment_id] = annotation

    def add_reddit_submission(self, source):
        self.submissions.append(RedditSubmission(RedditAnnotation(source, is_source=True)))

    def add_submission_branch(self, branch, sub_sample=False):
        annotation_branch = []
        branch_tokens = []
        class_comments = 0
        # First, convert to Python objects
        for annotation in branch:
            annotation = RedditAnnotation(annotation)
            if self.sdqc_to_int[annotation.sdqc_submission] == 3:
                class_comments += 1
            branch_tokens.extend(annotation.tokens)
            annotation_branch.append(annotation)

        # Filter out branches with pure commenting class labels
        if sub_sample and class_comments == len(branch):
            return

        # Compute cosine similarity
        source = self.submissions[self.last_submission()].source
        prev = source
        for annotation in annotation_branch:
            if annotation.comment_id not in self.annotations:  # Skip repeated annotations
                compute_similarity(annotation, prev, source, branch_tokens)
                self.analyse_annotation(annotation)  # Analyse relevant annotations
                self.annotations[annotation.comment_id] = annotation
                self.anno_to_branch_tokens[annotation.comment_id] = branch_tokens
                self.anno_to_prev[annotation.comment_id] = prev
                self.anno_to_source[annotation.comment_id] = source
            prev = annotation
        self.submissions[self.last_submission()].add_annotation_branch(annotation_branch)  # This might be unnecessary

        # if super_sample:
        #     prev = source
        #     for annotation in annotation_branch:
        #         if not self.sdqc_to_int[annotation.sdqc_submission] == 3:  # not commenting class
        #             annotation_copy = copy.deepcopy(annotation)
        #             annotation_copy.alter_id_and_text(words_to_replace=super_sample, early_stop=True)
        #             compute_similarity(annotation_copy, prev, source, branch_tokens)  # compare to original branch
        #             if annotation_copy.comment_id not in self.annotations:  # Skip repeated annotations
        #                 self.analyse_annotation(annotation_copy)  # Analyse relevant annotations
        #                 self.annotations[annotation_copy.comment_id] = annotation_copy
        #         prev = annotation

    def train_test_split(self, test_size=0.25, rand_state=42, shuffle=True, stratify=True):
        X = []
        y = []
        for annotation in self.iterate_annotations():
            X.append(annotation)
            y.append(self.sdqc_to_int[annotation.sdqc_submission])
        print('Splitting with test size', test_size)
        X_train, X_test, _, _ = train_test_split(
            X, y, test_size=test_size, random_state=rand_state, shuffle=shuffle, stratify=(y if stratify else None)
        )
        print('Train stats:')
        self.print_status_report(X_train)
        print('Test stats:')
        self.print_status_report(X_test)
        return X_train, X_test

    def super_sample(self, annotations, word_to_replace=5, early_stop=True):
        super_sample = []
        for i, annotation in enumerate(annotations):
            if not self.sdqc_to_int[annotation.sdqc_submission] == 3:
                annotation_copy = copy.deepcopy(annotation)
                annotation_copy.alter_id_and_text(words_to_replace=word_to_replace, early_stop=early_stop)
                compute_similarity(
                    annotation_copy,
                    self.anno_to_prev[annotation.comment_id],
                    self.anno_to_source[annotation.comment_id],
                    self.anno_to_branch_tokens[annotation.comment_id]
                )
                self.analyse_annotation(annotation_copy)
                self.annotations[annotation_copy.comment_id] = annotation_copy
                super_sample.append(annotation_copy)
            if i % 10 == 0:
                print('  %d' % i)
        annotations.extend(super_sample)
        return annotations


    def print_status_report(self, annotations=None):
        histogram = {
            0: 0,
            1: 0,
            2: 0,
            3: 0
        }
        ix_to_sdqc = {0: 'S', 1:'D', 2: 'Q', 3: 'C'}
        n = 0
        for annotation in (self.iterate_annotations() if not annotations else annotations):
            histogram[self.sdqc_to_int[annotation.sdqc_submission]] += 1
            n += 1
        print('Number of data points:', n)
        print('SDQC distribution:')
        for label, count in histogram.items():
            print('{}: {:4d} ({:.3f})'.format(ix_to_sdqc[label], count, float(count)/float(n)))

    def analyse_annotation(self, annotation):
        if not annotation:
            return
        self.handle(self.min_max['karma'], annotation.user_karma)
        self.handle(self.min_max['txt_len'], len(annotation.text))
        self.handle(self.min_max['afinn_score'], get_afinn_sentiment(annotation.text))
        self.handle(self.min_max['url_count'], annotation.tokens.count('urlurlurl'))
        self.handle(self.min_max['quote_count'], annotation.tokens.count('refrefref'))
        self.handle(self.min_max['cap_sequence_max_len'], len(max(re.findall(r"[A-ZÆØÅ]+", annotation.text), key=len, default='')))
        self.handle(self.min_max['tripDotCount'], annotation.text.count('...'))
        self.handle(self.min_max['q_mark_count'], annotation.text.count('?'))
        self.handle(self.min_max['e_mark_count'], annotation.text.count('!'))
        self.handle(self.min_max['cap_count'], sum(1 for c in annotation.text if c.isupper()))
        
        word_len = len(annotation.tokens)
        if not word_len == 0:
            self.handle(self.min_max['tokens_len'], word_len)
            self.handle(self.min_max['avg_word_len'],
                        sum([len(word) for word in annotation.tokens]) / word_len)
        self.handle(self.min_max['upvotes'], annotation.upvotes)
        self.handle(self.min_max['reply_count'], annotation.reply_count)
        self.handle_frequent_words(annotation)
        self.handle_bow(annotation.tokens)
        self.handle_ngram(annotation, self.freq_tri_gram, 3)

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

    def handle_bow(self, annotation_tokens):
        for t in annotation_tokens:
            self.bow.add(t)

    def handle_ngram(self, annotation, gram_dict, gram_size):
        annotation_tokens = annotation.tokens
        label = self.sdqc_to_int[annotation.sdqc_submission]
        label_dict = gram_dict[label]
        for (idx, t) in enumerate(annotation_tokens):
            if idx + gram_size < len(annotation_tokens)-1:
                seq = " ".join(annotation_tokens[idx:idx+gram_size])
                if seq in label_dict:
                    label_dict[seq] = label_dict[seq] + 1
                else:
                    label_dict[seq] = 1
        

    def get_frequent_words(self, take_count):
        histogram = {}
        word_count = {}
        for idx in range(len(self.freq_histogram)):
            keys = [(self.freq_histogram[idx][key], key) for key in self.freq_histogram[idx].keys()]
            keys.sort()
            keys.reverse()

            histogram[idx] = keys[:take_count]

            for (freq, word) in histogram[idx]:
                if word in word_count:
                    word_count[word] = word_count[word] + 1
                else:
                    word_count[word] = 1

        unique_histograms = {0: [], 1: [], 2: [], 3: []}

        for key, values in histogram.items():
            for (freq, word) in values:
                if word_count[word] == 4:
                    continue
                unique_histograms[key].append(word)

        return unique_histograms

    def iterate_annotations(self):
        for anno_id, annotation in self.annotations.items():
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
        return len(self.annotations)
