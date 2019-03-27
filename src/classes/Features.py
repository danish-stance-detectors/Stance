import word_embeddings
from classes.Annotation import RedditDataset

import re # regular expression

# Module for extracting features from comment annotations

class FeatureExtractor:

    def __init__(self, dataset, swear_words,
                 negation_words, negative_smileys, positive_smileys, wv_model, emb_dim, test=False):
        # using passed annotations if not testing
        if test:
            self.dataset = RedditDataset()
        else:
            self.dataset = dataset

        self.swear_words = swear_words
        self.negative_smileys = negative_smileys
        self.positive_smileys = positive_smileys
        self.negation_words = negation_words
        self.wv_model = wv_model
        self.emb_dim = emb_dim
        self.bow_words = set()

        self.sdqc_to_int = {
            "Supporting": 0,
            "Denying": 1,
            "Querying": 2,
            "Commenting": 3
        }
    
    def create_feature_vector_test(self, annotation):
        self.dataset.add_annotation(annotation)
        return self.create_feature_vector(annotation, include_reddit_features=False)

    def create_feature_vectors(self):
        self.make_bow_list()  # TODO: Maybe refactor to RedditDataset?
        feature_vectors = []
        for source, branch in self.dataset.iterate_branches():
            prev = None
            for annotation in branch:
                instance = self.create_feature_vector(source, branch, prev, annotation)
                feature_vectors.append(instance)
                prev = annotation
        return feature_vectors

    # Extracts features from comment annotation and extends the different kind of features to eachother.
    def create_feature_vector(self, source, branch, prev, comment, include_reddit_features=True):
        feature_vec = list()

        feature_vec.extend(self.text_features(comment.text, comment.tokens))

        # reddit specific features
        if include_reddit_features:
            feature_vec.extend(self.user_features(comment))
            feature_vec.extend(self.reddit_comment_features(comment))

        feature_vec.extend(self.special_words_in_text(comment.tokens, comment.text, self.swear_words, self.negation_words, self.negative_smileys, self.positive_smileys))
        feature_vec.extend(self.most_frequent_words_for_label(comment.tokens))
        
        feature_vec.extend(self.get_bow_presence(comment.tokens))

        if self.wv_model:
            avg_wembs = word_embeddings.avg_word_emb(comment.tokens, self.emb_dim, self.wv_model)
            simToSrc = self.cosine_similarity(comment.tokens, source.tokens)
            simToPrev = self.cosine_similarity(comment.tokens, prev.tokens) if prev else 0
            allTokens = []
            for annotation in branch:
                allTokens.extend(annotation.tokens)
            simToBranch = self.cosine_similarity(comment.tokens, allTokens)
            feature_vec.extend([simToSrc, simToPrev, simToBranch])
            feature_vec.extend(avg_wembs)
        parent_sdqc = self.sdqc_to_int[comment.sdqc_parent]
        sub_sdqc = self.sdqc_to_int[comment.sdqc_submission]

        return (comment.comment_id, parent_sdqc, sub_sdqc, feature_vec)

    def cosine_similarity(self, one, other):
        # Lookup words in w2c vocab
        words = []
        for token in one:
            if token in self.wv_model.vocab:  # check that the token exists
                words.append(token)
        other_words = []
        for token in other:
            if token in self.wv_model.vocab:
                other_words.append(token)

        if len(words) > 0 and len(other_words) > 0:  # make sure there is actually something to compare
            # cosine similarity between two sets of words
            return self.wv_model.n_similarity(other_words, words)
        else:
            return 0.  # no similarity if one set contains 0 words

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

        url_count = tokens.count('url')

        quote_count = tokens.count('Reference')
        
        # longest sequence of capital letters, default empty for 0 length
        cap_sequence_max_len = len(max(re.findall(r"[A-Z]+", text), key=len, default=''))

        # TODO: Normalize the following?
        # dotdotdot count
        # tripDotCount = text.count('...')

        # # Question mark count
        # q_mark_count = text.count('?')

        # # Exclamation mark count
        # e_mark_count = text.count('!')

        # Ratio of capital letters
        cap_count = sum(1 for c in text if c.isupper())
        cap_ratio = float(cap_count) / float(len(text)) if len(text) > 0 else 0.0
        return [period,
                e_mark,
                q_mark,
                hasTripDot,
                url_count,
                quote_count,
                # tripDotCount,
                # q_mark_count,
                # e_mark_count,
                cap_ratio,
                txt_len,
                tokens_len,
                avg_word_len,
                cap_sequence_max_len]


    def user_features(self, comment):
        karma_norm = self.normalize(comment.user_karma, 'karma')
        return [karma_norm, int(comment.user_gold_status), int(comment.user_is_employee), int(comment.user_has_verified_email)]


    # TODO: find special word cases
    # TODO: fix case where 'http://...' triggers a negative smiley (:/)
    def special_words_in_text(self, tokens, text, swear_words, negation_words, negative_smileys, positive_smileys):
        swear_count = self.count_lexicon_occurence(tokens, swear_words)
        negation_count = self.count_lexicon_occurence(tokens, negation_words)
        positive_smiley_count = self.count_lexicon_occurence(text.split(), positive_smileys)
        negative_smiley_count =  self.count_lexicon_occurence(text.split(), negative_smileys)

        return [swear_count, negation_count, positive_smiley_count, negative_smiley_count] #TODO: Normalize

    def reddit_comment_features(self, comment):
        upvotes_norm = self.normalize(comment.upvotes, 'upvotes')
        reply_count_norm = self.normalize(comment.reply_count, 'reply_count')
        return [upvotes_norm, reply_count_norm, int(comment.is_submitter)]

    
    def most_frequent_words_for_label(self, tokens):
        # self.annotations.make_frequent_words() must have been called for this to work
        
        vec = set()

        for histogram in self.dataset.get_frequent_words():
            for kv in histogram:
                vec.add(int(kv[1] in tokens))
 
        return vec

    # Gets BOW presence (binary) for tokens
    def get_bow_presence(self, tokens):
        return [1 if w in tokens else 0 for w in self.bow_words]

    ### HELPER METHODS ###

    # Counts the amount of words which appear in the lexicon
    def count_lexicon_occurence(self, words, lexion):
        return sum([1 if word in lexion else 0 for word in words])

    def normalize(self, x_i, property):
        min_x = self.dataset.get_min(property)
        max_x = self.dataset.get_max(property)
        if max_x-min_x != 0:
            return (x_i-min_x)/(max_x-min_x)
        
        return x_i

    # TODO: Refactor to RedditDataset to make BOW dynamically?
    def make_bow_list(self):
        words = []
        for a in self.dataset.annotations:
            for t in a.tokens:
                words.append(t)
        
        # turning it into set to get unique entries only
        self.bow_words = set(words)

    ### END OF HELPER METHODS ###