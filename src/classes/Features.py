import word_embeddings
from classes.Annotation import CommentAnnotation
from classes.Annotation import Annotations

# Module for extracting features from comment annotations

class FeatureExtractor:

    def __init__(self, annotations, swear_words, negation_words, negative_smileys, positive_smileys, wembs, emb_dim, test=False):
        #using passed annotations if not testing
        if test:
            self.annotations =  Annotations()
        else:
            self.annotations = annotations

        self.swear_words = swear_words
        self.negative_smileys = negative_smileys
        self.positive_smileys = positive_smileys
        self.negation_words = negation_words
        self.wembs = wembs
        self.emb_dim = emb_dim

        self.sdqc_to_int = {
            "Supporting": 0,
            "Denying": 1,
            "Querying": 2,
            "Commenting": 3
        }
    
    def create_feature_vector(self, annotation):
        self.annotations.add_annotation(annotation)
        return self.create_feature_vector(annotation, include_reddit_features=False)

    def create_feature_vectors(self):
        feature_vectors = []
        for annotation in self.annotations.iterate():
            instance = self.create_feature_vector(annotation)
            feature_vectors.append(instance)
        return feature_vectors

    # Extracts features from comment annotation and extends the different kind of features to eachother.
    def create_feature_vector(self, comment, include_reddit_features=True):
        feature_vec = list()

        wembs = word_embeddings.avg_word_emb(comment.tokens, self.emb_dim, self.wembs) if self.wembs else []

        feature_vec.extend(self.text_features(comment.text, comment.tokens))

        # reddit specific features
        if include_reddit_features:
            feature_vec.extend(self.user_features(comment))
            feature_vec.extend(self.reddit_comment_features(comment))

        feature_vec.extend(self.special_words_in_text(comment.tokens, comment.text, self.swear_words, self.negation_words, self.negative_smileys, self.positive_smileys))
        
        if wembs:
            feature_vec.extend(wembs)
        parent_sdqc = self.sdqc_to_int[comment.sdqc_parent]
        sub_sdqc = self.sdqc_to_int[comment.sdqc_submission]

        return (comment.comment_id, parent_sdqc, sub_sdqc, feature_vec)

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
        q_mark = int('?' in text)
        # Edit in text
        edited = int('Edit:' in text)

        # dotdotdot
        hasTripDot = int('...' in text)

        url_count = tokens.count('url')

        quote_count = tokens.count('Reference')

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
                avg_word_len]


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

    # Counts the amount of words which appear in the lexicon
    def count_lexicon_occurence(self, words, lexion):
        return sum([1 if word in lexion else 0 for word in words])

    def reddit_comment_features(self, comment):
        upvotes_norm = self.normalize(comment.upvotes, 'upvotes')
        reply_count_norm = self.normalize(comment.reply_count, 'reply_count')
        return [upvotes_norm, reply_count_norm, int(comment.is_submitter)]

    def normalize(self, x_i, property):
        min_x = self.annotations.get_min(property)
        max_x = self.annotations.get_max(property)
        if max_x-min_x != 0:
            return (x_i-min_x)/(max_x-min_x)
        
        return x_i