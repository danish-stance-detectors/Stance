# external imports
import json
import time
import os
import csv
import sys
from datetime import datetime
import argparse

### Data loading file for semeval label and time data ###

# Paths to data
#training_data_path = '../data/semeval_rumour_data/semeval2017-task8-dataset/rumoureval-data/'
training_data_path_en = '../data/pheme_data/pheme-rumour-scheme-dataset/threads/en/'
training_data_path_de = '../data/pheme_data/pheme-rumour-scheme-dataset/threads/de/'
training_data_path_en_old = '../data/pheme_data/old_version/pheme-rumour-scheme-dataset/threads/en/'
training_data_path_de_old = '../data/pheme_data/old_version/pheme-rumour-scheme-dataset/threads/de/'
test_data_path = '../data/semeval_rumour_data/semeval2017-task8-test-labels/'

training_labels_path = '../data/semeval_rumour_data/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-train.json'
dev_labels_path = '../data/semeval_rumour_data/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-dev.json'
test_stance_labels = '../data/semeval_rumour_data/semeval2017-task8-test-labels/subtaska.json'
pheme_stance_labels_de = '../data/pheme_data/pheme-rumour-scheme-dataset/annotations/de-scheme-annotations.json'
pheme_stance_labels_en = '../data/pheme_data/pheme-rumour-scheme-dataset/annotations/en-scheme-annotations.json'
pheme_stance_labels_de_old = '../data/pheme_data/old_version/pheme-rumour-scheme-dataset/annotations/de-scheme-annotations.json'
pheme_stance_labels_en_old = '../data/pheme_data/old_version/pheme-rumour-scheme-dataset/annotations/en-scheme-annotations.json'

rumour_train_labels_path = '../data/semeval_rumour_data/semeval2017-task8-dataset/traindev/rumoureval-subtaskB-train.json'
rumour_dev_labels_path = '../data/semeval_rumour_data/semeval2017-task8-dataset/traindev/rumoureval-subtaskB-dev.json'
rumour_test_labels_path = '../data/semeval_rumour_data/semeval2017-task8-test-labels/subtaskb.json'

semeval_train_dev_data_path = '../data/semeval_rumour_data/semeval2017-task8-dataset/rumoureval-data/'
semeval_test_data_path = '../data/semeval_rumour_data/semeval2017-task8-test-data/'

data_folder = '../data/hmm/'

keep_time = False

def main(argv):

    parser = argparse.ArgumentParser(description='Preprocessing of data files from pheme and semeval for rumour veracity hmm classification')

    parser.add_argument('-f', '--data file path', dest='file', default='../data/semeval_rumour_data/semeval2017-task8-dataset/rumoureval-data/', help='Input folder holding annotated data')
    parser.add_argument('-o', '--out file path', dest='outfile', default='../data/hmm/hmm_data.csv', help='Output filer holding preprocessed data')
    parser.add_argument('-time', '--keep_time', action='store_true', default=False, help='Keep the normalized timestamps')
    parser.add_argument('-train', '--train', action='store_true', default=False, help='Include and write out semeval train data')
    parser.add_argument('-dev', '--dev', action='store_true', default=False, help='Include and write out semeval dev data')
    parser.add_argument('-test', '--test', action='store_true', default=False, help='Include and write out semeval test data as well')
    
    args = parser.parse_args(argv)

    # write train dev to separate file
    if args.train or args.dev:
        read_rumour_data_dev_train(args.file, args.outfile, args.train, args.dev, False, args.keep_time)
    
    # write test to separate file
    if args.test:
        read_rumour_data_dev_train(args.file, args.outfile.rstrip('.csv') + '_test.csv', False, False, args.test, args.keep_time)

# Events from which dungs 18 used rumour data from
dungs18_events = [
    'charliehebdo',
    'ferguson',
    'germanwings-crash',
    'ottawashooting',
    'sydneysiege'
]

sdqc_to_int = {
    'support' : 0,
    'deny'    : 1,
    'query'   : 2,
    'comment' : 3,
    'supporting': 0, # for pheme data set
    'agreed' : 0,
    'disagreed' : 1,
    'denying': 1,
    'appeal-for-more-information' : 2,
    'underspecified' : 3,
}

rumour_truth_to_int = {
    'true'       : 1,
    'unverified' : 0,
    'false'      : 0
}

# Reads labels from 'tweet_id' : 'label' json file list
def read_stance_labels(path):
    with open(path, 'r') as label_file:
        label_dict = dict()
        for tweet_id, sdqc_label in json.load(label_file).items():
            label_dict[tweet_id] = sdqc_to_int[sdqc_label]
        
        return label_dict

# reads rumour labels
def read_rumour_labels(path):
    with open(path, 'r') as label_file:
        label_dict = dict()
        for conv_id, truth_label in json.load(label_file).items():
            #if truth_label != 'unverified':
            label_dict[conv_id] = rumour_truth_to_int[truth_label]
    
    return label_dict

def read_pheme_labels(path):
    with open(path, 'r') as label_file:
        label_dict = dict()
        for line in label_file.readlines():
            if not line.startswith("#"):
                entry = json.loads(line)
                # top level and lower have different support names
                if 'support' in entry:
                    label_dict[entry['tweetid']] = sdqc_to_int[entry['support']]
                else:
                    label_dict[entry['tweetid']] = sdqc_to_int[entry['responsetype-vs-source']]
    
    return label_dict

def read_all_rumours(path, rumour_dict, stance_dict, include_all=False, keep_time_=False):

    events = os.listdir(path)
    data = []

    for event in events:
        if event in dungs18_events:
            # get all conversations in event
            event_path = os.path.join(path, event)
            event_data = read_conversations_in_dir(event_path, event, rumour_dict, stance_dict, min_len=1, keep_time=keep_time_)

            # Event must yield atleast 5 conversations with 5 or more tweets
            # tmp_len = len([x for x in event_data if len(x[2]) > 4])
            # if tmp_len > 4 or include_all:
                #event_data = [x for x in event_data if len(x[2]) >= 10]
            data.extend(event_data)

    return data


unpack_tupples = lambda l : [x for t in l for x in t]

def read_test_rumours(path, rumour_dict, stance_dict):
    return read_conversations_in_dir(path, 'test', rumour_dict, stance_dict)

# reads all conversations in a specific path folder
def read_conversations_in_dir(path, event, rumour_dict, stance_dict, min_len=1, keep_time=False):
    conversations = os.listdir(path)
    event_data = []
    for conv in conversations:
        if conv in rumour_dict:
            replies = []
            folder = os.path.join(path, conv)

            src_tweet_path = ''
            if os.path.exists(folder + "/source-tweet"):
                src_tweet_path = os.path.join(folder, "source-tweet")
            else:
                src_tweet_path = os.path.join(folder, "source-tweets")
            
            src_tweet = os.listdir(src_tweet_path)[0] #There is only one
            
            replies.append(read_tweet_id_time(os.path.join(src_tweet_path, src_tweet), stance_dict))

            replies_path = ''
            if os.path.exists(folder + "/replies"):
                replies_path = os.path.join(folder, "replies")
            else:
                replies_path = os.path.join(folder, "reactions")
            
            if (os.path.exists(replies_path)):
                replies_files = os.listdir(replies_path)
                for file in replies_files:
                    tweet = read_tweet_id_time(os.path.join(replies_path, file), stance_dict)
                    if tweet is not None:
                        replies.append(tweet)
            
            replies = sorted(replies, key = lambda x : x[1]) # sort replies by time ascending

            if keep_time: # keep time
                # normalize time
                max_time = max([x[1] for x in replies])
                min_time = min([x[1] for x in replies])
                
                for i in range(len(replies)):
                    if (max_time - min_time) == 0:
                        norm_time = 0.0
                    else:
                        norm_time = (replies[i][1] - min_time) / (max_time - min_time)
                    replies[i] = (replies[i][0], norm_time)
            if keep_time: # keep times
                replies = unpack_tupples(replies)
            else:
                replies = [x[0] for x in replies] # throw away time stamp

            if len(replies) >= min_len:
                event_data.append((event, rumour_dict[conv], replies))
    
    return event_data

# returns (id, created) at for tweet on path
def read_tweet_id_time(path, label_dict):
    with open(path, 'r') as file:
        tweet = json.load(file)

        created_at = parse_time_string(tweet['created_at'])
        tweet_id = str(tweet['id'])
        if tweet_id in label_dict:
            return label_dict[tweet_id], created_at
        elif tweet['id'] in label_dict:
            print("here")
        elif int(tweet['id']) in label_dict:
            print("int here")

# parses time string for example :"Wed Jan 07 11:07:51 +0000 2015"
# to timestamp
def parse_time_string(time_string):
    return time.mktime(time.strptime(time_string, "%a %b %d %H:%M:%S +0000 %Y"))

# Write data to file
def write_hmm_data(filename, data):
    if not data:
        return

    print('Writing hmm vectors to', filename)
    with open(filename, "w+", newline='') as out_file:
        csv_writer = csv.writer(out_file, delimiter='\t')
        csv_writer.writerow(['Event', 'TruthStatus', 'SDQC_Labels'])
        
        for (event, truth_status, labels) in data:
            csv_writer.writerow([event, truth_status, labels])
    print('Done')

# Reads rumour labels. Bool flags to control train, dev and test labels
def read_rumour_labels_all(train=False, dev=False, test=False):
    rumour_labels = dict()
    if train:
        rumour_labels.update(read_rumour_labels(rumour_train_labels_path))
    if dev:
        rumour_labels.update(read_rumour_labels(rumour_dev_labels_path))
    if test:
        rumour_labels.read_rumour_labels(rumour_test_labels_path)
    
    return rumour_labels

# Reads stance labels. Bool flags to control train, dev and test labels
def read_stance_labels_all(train=False, dev=False, test=False):
    stance_labels = dict()
    if train:
        stance_labels.update(read_stance_labels(training_labels_path))
    if dev:
        stance_labels.update(read_stance_labels(dev_labels_path))
    if test:
        stance_labels.update(read_stance_labels(test_stance_labels))
    
    return stance_labels

def read_rumour_data_dev_train(path, out_path, train_, dev_, test_, keep_time):

    rumour_labels = read_rumour_labels_all(train=train_, dev=dev_, test=test_)
    stance_labels = read_stance_labels_all(train=train_, dev=dev_, test=test_)

    rumour_data_train = read_all_rumours(semeval_train_dev_data_path, rumour_labels, stance_labels, keep_time_=keep_time)
    
    print("Found data for {} training rumours".format(len(rumour_data_train)))
    
    write_hmm_data(out_path, rumour_data_train)

#rumour labels
# rumour_labels = dict()
# rumour_test_labels = dict()
# stance_labels = dict()

# rumour_labels.update(read_rumour_labels(rumour_train_labels_path))
# rumour_labels.update(read_rumour_labels(rumour_dev_labels_path))

# rumour_test_labels = read_rumour_labels(rumour_test_labels_path)

# # stance labels
# stance_labels.update(read_stance_labels(training_labels_path))
# stance_labels.update(read_stance_labels(dev_labels_path))
# stance_labels.update(read_stance_labels(test_stance_labels))

# rumour_data_train= read_all_rumours(semeval_train_dev_data_path, rumour_labels, stance_labels)

# rumour_data_test = read_test_rumours(semeval_test_data_path, rumour_test_labels, stance_labels)

# print("Found data for {} training rumours".format(len(rumour_data_train)))

# write_hmm_data(data_folder + 'semeval_train_dev.csv', rumour_data_train)
# write_hmm_data(data_folder + 'semeval_test.csv', rumour_data_test)

if __name__ == "__main__":
    main(sys.argv[1:])