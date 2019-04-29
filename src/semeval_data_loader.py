# external imports
import json
import time
import os
from datetime import datetime

# own imports
from preprocess import write_hmm_data 

### Data loading file for semeval label and time data ###

# Paths to data
training_data = '../data/semeval_rumour_data/semeval2017-task8-dataset/rumoureval-data/'

training_labels = '../data/semeval_rumour_data/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-train.json'
dev_labels = '../data/semeval_rumour_data/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-dev.json'

rumour_labels = '../data/semeval_rumour_data/semeval2017-task8-dataset/traindev/rumoureval-subtaskB-train.json'
rumour_dev_labels = '../data/semeval_rumour_data/semeval2017-task8-dataset/traindev/rumoureval-subtaskB-dev.json'

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
            label_dict[int(tweet_id)] = sdqc_to_int[sdqc_label]
        
        return label_dict

# reads rumour labels
def read_rumour_labels(path):
        with open(path, 'r') as label_file:
            label_dict = dict()
            for conv_id, truth_label in json.load(label_file).items():
                label_dict[int(conv_id)] = rumour_truth_to_int[truth_label]
        
        return label_dict

def read_all_rumours(path, rumour_dict, stance_dict):

    events = os.listdir(path)
    data = []

    for event in events:
        if event in dungs18_events:
            conversations = os.listdir(os.path.join(path, event))
            event_data = []
            for conv in conversations:
                conv_i = int(conv)
                if conv_i in rumour_dict:
                    replies = []
                    folder = os.path.join(path, event, conv)
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
                    event_data.append((rumour_dict[conv_i], replies))
            
            # tmp_len = sum([1 for x in event_data if len(x[1]) >= 5])
            # if tmp_len >= 5:
            data.extend(event_data)

    return data

# returns (id, created) at for tweet on path
def read_tweet_id_time(path, label_dict):
    with open(path, 'r') as file:
        tweet = json.load(file)

        created_at = parse_time_string(tweet['created_at'])
        tweet_id = int(tweet['id'])

        return label_dict[tweet_id], created_at

# parses time string for example :"Wed Jan 07 11:07:51 +0000 2015"
# to timestamp
def parse_time_string(time_string):
    return time.mktime(time.strptime(time_string, "%a %b %d %H:%M:%S +0000 %Y"))


rumour_labels = read_rumour_labels(rumour_labels)
rumour_labels.update(read_rumour_labels(rumour_dev_labels)) # add dev labels as well

stance_labels = read_stance_labels(training_labels)
stance_labels.update(read_stance_labels(dev_labels))

rumour_data = read_all_rumours(training_data, rumour_labels, stance_labels)

write_hmm_data('semeval_rumours_dungs_unv.csv', rumour_data)