import word_embeddings
from classes.Annotation import RedditDataset
from classes.afinn_sentiment import get_afinn_sentiment
from classes.polyglot_pos import pos_tags_occurence
import re

# Module for extracting features from comment annotations


class FeatureExtractor:

    def __init__(self, dataset, test=False):
        # using passed annotations if not testing
        if test:
            self.dataset = RedditDataset()
        else:
            self.dataset = dataset

        self.swear_words = []
        self.negative_smileys = []
        self.positive_smileys = []
        self.negation_words = []
        self.bow_words = set()

        self.sdqc_to_int = {
            "Supporting": 0,
            "Denying": 1,
            "Querying": 2,
            "Commenting": 3
        }
    
    def create_feature_vector_test(self, annotation):
        self.dataset.add_annotation(annotation)
        return self.create_feature_vector(annotation, False, False, False, False, False, False, False, False)

    def create_feature_vectors(self, data, wembs, text, lexicon, sentiment, reddit, most_freq, bow, pos):
        feature_vectors = []
        for annotation in data:
            instance = self.create_feature_vector(
                annotation, wembs, text, lexicon, sentiment, reddit, most_freq, bow, pos
            )
            feature_vectors.append(instance)
        return feature_vectors

    # Extracts features from comment annotation and extends the different kind of features to eachother.
    def create_feature_vector(self, comment, wembs, text, lexicon, sentiment, reddit, most_freq, bow, pos):
        feature_vec = list()
        if text:
            feature_vec.append(self.text_features(comment.text, comment.tokens))
        if sentiment:
            feature_vec.append(self.normalize(get_afinn_sentiment(comment.text), 'afinn_score'))
        if lexicon:
            feature_vec.append(self.special_words_in_text(comment.tokens, comment.text))
        if reddit:
            feature_vec.append(self.user_features(comment))
            feature_vec.append(self.reddit_comment_features(comment))
        if most_freq:
            feature_vec.append(self.most_frequent_words_for_label(comment.tokens, most_freq))
        if bow:
            feature_vec.append(self.get_bow_presence(comment.tokens))
        if pos:
            feature_vec.append(pos_tags_occurence(comment.text))

        if wembs:
            feature_vec.extend([comment.sim_to_src, comment.sim_to_prev, comment.sim_to_branch])
            avg_wembs = word_embeddings.avg_word_emb(comment.tokens)
            feature_vec.extend(avg_wembs)
        parent_sdqc = self.sdqc_to_int[comment.sdqc_parent]
        sub_sdqc = self.sdqc_to_int[comment.sdqc_submission]

        return comment.comment_id, parent_sdqc, sub_sdqc, feature_vec

    def text_features(self, text, tokens):
        # number of chars
        txt_len = self.normalize(len(text), 'txt_len') if len(text) > 0 else 0
        # number of words
        tokens_len = 0
        avg_word_len = 0
        if len(tokens) > 0:
            tokens_len = self.normalize(len(tokens), 'tokens_len')
            avg_word_len_true = sum([len(word) for word in tokens]) / len(tokens)
            avg_word_len = self.normalize(avg_word_len_true,'avg_word_len')

        # Period (.)
        period = int('.' in text)
        # Exclamation mark (!)
        e_mark = int('!' in text)
        # Question mark(?)
        q_mark = int('?' in text or any(word.startswith('hv') for word in text.split()))
        # Edit in text
        edited = int('Edit:' in text)

        # dotdotdot
        hasTripDot = int('...' in text)

        url_count = self.normalize(tokens.count('urlurlurl'), 'url_count')

        quote_count = self.normalize(tokens.count('refrefref'), 'quote_count')
        
        # longest sequence of capital letters, default empty for 0 length
        cap_sequence_max_len = len(max(re.findall(r"[A-ZÆØÅ]+", text), key=len, default=''))  # TODO: Normalize?
        cap_sequence_max_len = self.normalize(cap_sequence_max_len, 'cap_sequence_max_len')
        
        # dotdotdot count
        tripDotCount = self.normalize(text.count('...'), 'tripDotCount')

        # Question mark count
        q_mark_count = self.normalize(text.count('?'), 'q_mark_count')

        # Exclamation mark count
        e_mark_count = self.normalize(text.count('!'), 'e_mark_count')

        # Ratio of capital letters
        cap_count = self.normalize(sum(1 for c in text if c.isupper()), 'cap_count')

        cap_ratio = float(cap_count) / float(len(text)) if len(text) > 0 else 0.0
        return [period,
                e_mark,
                q_mark,
                hasTripDot,
                url_count,
                quote_count,
                tripDotCount,
                q_mark_count,
                e_mark_count,
                cap_ratio,
                txt_len,
                tokens_len,
                avg_word_len,
                cap_sequence_max_len]

    def user_features(self, comment):
        karma_norm = self.normalize(comment.user_karma, 'karma')
        return [karma_norm, int(comment.user_gold_status), int(comment.user_is_employee), int(comment.user_has_verified_email)]

    # TODO: find special word cases
    def special_words_in_text(self, tokens, text):
        if not self.positive_smileys:
            self.positive_smileys = read_lexicon('../data/lexicon/positive_smileys.txt')
        if not self.negative_smileys:
            self.negative_smileys = read_lexicon('../data/lexicon/negative_smileys.txt')
        if not self.swear_words:
            self.swear_words = read_lexicon('../data/lexicon/swear_words.txt')
        if not self.negation_words:
            self.negation_words = read_lexicon('../data/lexicon/negation_words.txt')

        swear_count = self.count_lexicon_occurence(tokens, self.swear_words)
        negation_count = self.count_lexicon_occurence(tokens, self.negation_words)
        positive_smiley_count = self.count_lexicon_occurence(text.split(), self.positive_smileys)
        negative_smiley_count =  self.count_lexicon_occurence(text.split(), self.negative_smileys)

        return [swear_count, negation_count, positive_smiley_count, negative_smiley_count]  # TODO: Normalize

    def reddit_comment_features(self, comment):
        upvotes_norm = self.normalize(comment.upvotes, 'upvotes')
        reply_count_norm = self.normalize(comment.reply_count, 'reply_count')
        return [upvotes_norm, reply_count_norm, int(comment.is_submitter)]

    def most_frequent_words_for_label(self, tokens, n_most=50):
        # self.annotations.make_frequent_words() must have been called for this to work
        
        vec = []

        histograms = self.dataset.get_frequent_words(n_most)
        for sdqc_id, histogram in histograms.items():
            for count, freq_token in histogram:
                vec.append(int(freq_token in tokens))
 
        return vec

    # Gets BOW presence (binary) for tokens
    def get_bow_presence(self, tokens):
        return [1 if w in tokens else 0 for w in self.dataset.bow]

    ### HELPER METHODS ###

    # Counts the amount of words which appear in the lexicon
    def count_lexicon_occurence(self, words, lexion):
        return sum([1 if word in lexion else 0 for word in words])

    def normalize(self, x_i, prop):
        min_x = self.dataset.get_min(prop)
        max_x = self.dataset.get_max(prop)
        if max_x-min_x != 0:
            return (x_i-min_x)/(max_x-min_x)
        
        return x_i


def read_lexicon(file_path):  # TODO: Maybe make class method?
    """Loads lexicon file given path. Assumes file has one word per line"""
    with open(file_path, "r", encoding='utf8') as lexicon_file:
        return [line.strip().lower() for line in lexicon_file.readlines()]

    ### END OF HELPER METHODS ###