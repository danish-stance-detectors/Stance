# external imports
import json
import time
import os
import csv
from datetime import datetime

### Data loading file for semeval label and time data ###

# Paths to data
training_data_path = '../data/semeval_rumour_data/semeval2017-task8-dataset/rumoureval-data/'
test_data_path = '../data/semeval_rumour_data/semeval2017-task8-test-labels/'

training_labels_path = '../data/semeval_rumour_data/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-train.json'
dev_labels_path = '../data/semeval_rumour_data/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-dev.json'
test_stance_labels = '../data/semeval_rumour_data/semeval2017-task8-test-labels/subtaska.json'

rumour_train_labels_path = '../data/semeval_rumour_data/semeval2017-task8-dataset/traindev/rumoureval-subtaskB-train.json'
rumour_dev_labels_path = '../data/semeval_rumour_data/semeval2017-task8-dataset/traindev/rumoureval-subtaskB-dev.json'
rumour_test_labels_path = '../data/semeval_rumour_data/semeval2017-task8-test-labels/subtaskb.json'

data_folder = '../data/hmm/'

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
    'comment' : 3
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

def read_all_rumours(path, rumour_dict, stance_dict, include_all=False):

    events = os.listdir(path)
    data = []

    for event in events:
        if event in dungs18_events:
            # get all conversations in event
            event_path = os.path.join(path, event)
            event_data = read_conversations_in_dir(event_path, event, rumour_dict, stance_dict, min_len=10)
            
            # Event must yield atleast 5 conversations with 5 or more tweets
            tmp_len = len([x for x in event_data if len(x[2]) > 4])
            if tmp_len > 4 or include_all:
                data.extend(event_data)

    return data

def read_test_rumours(path, rumour_dict, stance_dict):
    return read_conversations_in_dir(path, 'test', rumour_dict, stance_dict)

# reads all conversations in a specific path folder
def read_conversations_in_dir(path, event, rumour_dict, stance_dict, min_len=1):
    conversations = os.listdir(path)
    event_data = []
    for conv in conversations:
        if conv in rumour_dict:
            replies = []
            folder = os.path.join(path, conv)
            src_tweet_path = os.path.join(folder, "source-tweet")
            src_tweet = os.listdir(src_tweet_path)[0] #There is only one
            replies.append(read_tweet_id_time(os.path.join(src_tweet_path, src_tweet), stance_dict))

            replies_path = os.path.join(folder, "replies")
            
            if (os.path.exists(replies_path)):
                replies_files = os.listdir(replies_path)
                for file in replies_files:
                    replies.append(read_tweet_id_time(os.path.join(replies_path, file), stance_dict))
            
            replies = sorted(replies, key = lambda x : x[1]) # sort replies by time ascending
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

        return label_dict[tweet_id], created_at

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

#rumour labels
rumour_labels = dict()
stance_labels = dict()

rumour_labels.update(read_rumour_labels(rumour_train_labels_path))
rumour_labels.update(read_rumour_labels(rumour_dev_labels_path))
# rumour_labels.update(read_rumour_labels(rumour_test_labels_path))

# stance labels
stance_labels.update(read_stance_labels(training_labels_path))
stance_labels.update(read_stance_labels(dev_labels_path))
# stance_labels.update(read_stance_labels(test_stance_labels))

rumour_data = read_all_rumours(training_data_path, rumour_labels, stance_labels, include_all=False)
#rumour_data_dev = read_all_rumours(training_data_path, rumour_labels_test, stance_labels_test)
#rumour_data_test = read_test_rumours(test_data_path, rumour_labels_test, stance_labels_test)

#rumour_data_train.extend(rumour_data_dev)
#rumour_data_train.extend(rumour_data_test)

print("Found data for {} training rumours".format(len(rumour_data)))

write_hmm_data(data_folder + 'semeval_rumours_train.csv', rumour_data)