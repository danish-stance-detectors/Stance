import os, json, csv, sys
import word_embeddings
from classes.Annotation import RedditDataset
from classes.Features import FeatureExtractor

datafolder = '../data/'
preprocessed_folder = os.path.join(datafolder, 'preprocessed/')
annotated_folder = os.path.join(datafolder, 'annotated/')
fasttext_data = os.path.join(datafolder, 'fasttext/cc.da.300.bin')
word2vec_data = os.path.join(datafolder, 'word2vec/dsl_sentences_300_cbow_negative.kv')

# Loads lexicon file given path
# Assumes file has one word per line
def read_lexicon(file_path):
    with open(file_path, "r") as lexicon_file:
        return [line.strip().lower() for line in lexicon_file.readlines()]

swear_words = []
negation_words = []
positive_smileys = read_lexicon('../data/lexicon/positive_smileys.txt')
negative_smileys = read_lexicon('../data/lexicon/negative_smileys.txt')

with open('../data/lexicon/swear_words.txt', "r") as swear_word_file:
    for line in swear_word_file.readlines():
        swear_words.append(line.strip().lower())

with open('../data/lexicon/negation_words.txt', "r") as negation_word_file:
    for line in negation_word_file.readlines():
        negation_words.append(line.strip().lower())


def loadAnnotations(filename):
    if not filename:
        return
    dataset = RedditDataset()
    for rumour_folder in os.listdir(filename)[:1]:
        rumour_folder_path = os.path.join(filename, rumour_folder)
        if not os.path.isdir(rumour_folder_path):
            continue
        for submission_json in os.listdir(rumour_folder_path):
            submission_json_path = os.path.join(rumour_folder_path, submission_json)
            with open(submission_json_path, "r", encoding='utf-8') as file:
                json_obj = json.load(file)
                sub = json_obj['redditSubmission']
                dataset.add_reddit_submission(sub)
                branches = json_obj['branches']
                for branch in branches:
                    dataset.add_submission_branch(branch)
    return dataset


def preprocess(annotations, wv_model=None, emb_dim=100):
    if not annotations:
        return
    feature_extractor = \
        FeatureExtractor(annotations, swear_words, negation_words, negative_smileys, positive_smileys, wv_model, emb_dim)
    return feature_extractor.create_feature_vectors()


def write_preprocessed(preprocessed_data, filename):
    if not preprocessed_data:
        return
    with open(preprocessed_folder + filename, "w+", newline='') as out_file:
        csv_writer = csv.writer(out_file, delimiter='\t')
        csv_writer.writerow(['comment_id', 'sdqc_parent', 'sdqc_submission', 'feature_vector'])
        
        for (id, sdqc_p, sdqc_s, vec) in preprocessed_data:
            csv_writer.writerow([id, sdqc_p, sdqc_s, vec])

# def get_branches(annotation_list):
#     # comments with no replies are bottom level
#     bottom_annotations = list(filter(lambda x: x.reply_count == 0, annotation_list))
#
#     #get branch of each bottom level comment
#     branches = [get_branch_of_bottom(x, annotation_list) for x in bottom_annotations]
#     return branches
#
# # used by 'get_branches' to get all branches from a list of annotations
# def get_branch_of_bottom(bottom_level_comment, annotation_list):
#     # return empty list if deleted
#     if bottom_level_comment.is_deleted:
#         return []
#
#     # return list containing itself is it is top level comment
#     parent_id = bottom_level_comment.parent_id
#     if parent_id == bottom_level_comment.submission_id:
#         return [bottom_level_comment]
#
#     branch = []
#     current_comment = bottom_level_comment
#     # continue until top level
#     while current_comment.parent_id != current_comment.submission_id:
#         branch.append(current_comment)
#         # set current comment to parent
#         current_comment = [x for x in annotation_list if x.comment_id == current_comment.parent_id][0]
#
#     # this is top level comment, add
#     branch.append(current_comment)
#     return branch

def main(argv):
    # data_folder = wembs = emb_dim = None
    # try:
    #     opts, _ = getopt.getopt(argv, "l:d:e:", ["loadw2v=","data=","emb_dim=","help"])
    # except getopt.GetoptError:
    #     print("see: preprocess.py -help")
    #     sys.exit(2)
    # for opt, arg in opts:
    #     if opt in ('-l', '-loadw2v'):
    #         wembs = word_embeddings.load_saved_word2vec_wv(arg)
    #     elif opt in ('-d', '-data'):
    #         data_folder = arg
    #     elif opt in ('-e', '-emb_dim'):
    #         emb_dim = arg
    
    wv_model = word_embeddings.load_saved_word2vec_wv(word2vec_data)
    annotations = loadAnnotations(annotated_folder)
    
    data = preprocess(annotations, wv_model, emb_dim=300)
    write_preprocessed(data, 'preprocessed_test_bow.csv')

if __name__ == "__main__":
    main(sys.argv[1:])