import os, json, csv, sys, time
import argparse


hmm_datafile = '../data/hmm/preprocessed_hmm.csv'
semeval_hmm_data = '../data/hmm/semeval_rumours_train.csv'
tab = '\t'

def main(argv):
    
    parser = argparse.ArgumentParser(description='Preprocessing of data files for rumour veracity hmm classification')
    parser.add_argument('-hmm', '--hiddenMarkovModel', default=False, dest='hmm', action='store_true', help='Get HMM features instead of stance preprocessing features')
    parser.add_argument('-br', '--branch', default=False, action='store_true', help='Get hmm features in branches')
    parser.add_argument('-cmt', '--comment', default=False, action='store_true', help='Get hmm features in comment trees')
    parser.add_argument('-t', '--time', default=False, action='store_true', help='Get hmm features with both stance and time')
    parser.add_argument('-f', '--data file path', dest='file', default='../data/annotated/', help='Input folder holding annotated data')
    parser.add_argument('-o', '--out file path', dest='outfile', default='../data/hmm/hmm_data.csv', help='Output filer holding preprocessed data')
    args = parser.parse_args(argv)

    data = []
    
    if args.hmm:
        data = read_hmm_data_no_branches(args.file, args.time)
    elif args.branch:
        data = read_hmm_data(args.file, args.time)
    elif args.comment:
        data = read_hmm_data_cmt_trees(args.file, args.time)
    
    write_hmm_data(args.outfile, data)

# unpack_tupples = lambda l : [x for t in l for x in t]

def unpack_tupples(list):
    flat_list = []
    for item in list:
        if type(item) == list:
            flat_list.extend(item)
        else:
            flat_list.append(item)
    
    return flat_list

rumour_truth_to_int = {
    'False' : 0,
    'True'  : 1,
    'Unverified' : 1
}

def read_hmm_data(filename, keep_time):
    if not filename:
        return
    
    label_data = []
    sdqc_to_int = {'Supporting':0, 'Denying':1, 'Querying':2, 'Commenting':3}
    label_distribution = {'Supporting':0, 'Denying':0, 'Querying':0, 'Commenting':0}
    rumour_count = 0
    truth_count = 0
    for rumour_folder in os.listdir(filename):
        rumour_folder_path = os.path.join(filename, rumour_folder)
        if not os.path.isdir(rumour_folder_path):
            continue
        for submission_json in os.listdir(rumour_folder_path):
            submission_json_path = os.path.join(rumour_folder_path, submission_json)
            with open(submission_json_path, "r", encoding='utf-8') as file:
                json_obj = json.load(file)
                sub = json_obj['redditSubmission']
                if sub['IsRumour'] and not sub['IsIrrelevant']:
                    print("Adding {} with id {} as rumour".format(sub['title'], submission_json))
                    rumour_truth = rumour_truth_to_int[sub['TruthStatus']]
                    rumour_count += 1
                    truth_count += rumour_truth
                    for branch in json_obj['branches']:
                        branch_labels = []
                        for comment in branch:
                            label = comment['comment']['SDQC_Submission']
                            label_distribution[label] += 1
                            if keep_time:
                                created = comment['comment']['created']
                                time_stamp = time.mktime(time.strptime(created, "%Y-%m-%dT%H:%M:%S"))
                                branch_labels.append((sdqc_to_int[label], time_stamp))
                            else:
                                branch_labels.append(sdqc_to_int[label])
                            
                        if keep_time:
                            # normalize time
                            max_time = max([x[1] for x in branch_labels])
                            min_time = min([x[1] for x in branch_labels])
                            
                            for i in range(len(branch_labels)):
                                if (max_time - min_time) == 0:
                                    norm_time = 0.0
                                else:
                                    norm_time = (branch_labels[i][1] - min_time) / (max_time - min_time)
                                branch_labels[i] = (branch_labels[i][0], norm_time)

                            label_data.append((rumour_truth, unpack_tupples(branch_labels)))
                        else:
                            label_data.append((rumour_truth, branch_labels))
    
    print("Preprocessed {} rumours of which {} were true".format(rumour_count, truth_count))
    print("With sdqc overall distribution: ")
    print(label_distribution)
    return label_data

def read_hmm_data_no_branches(filename, keep_time):
    if not filename:
        print("Cannot run method read_hmm_data_no_branches without filename parameter")
        return
    
    label_data = []

    sdqc_to_int = {'Supporting':0, 'Denying':1, 'Querying':2, 'Commenting':3}
    label_distribution = {'Supporting':0, 'Denying':0, 'Querying':0, 'Commenting':0}
    rumour_count = 0
    truth_count = 0
    for rumour_folder in os.listdir(filename):
        rumour_folder_path = os.path.join(filename, rumour_folder)
        if not os.path.isdir(rumour_folder_path):
            continue
        for submission_json in os.listdir(rumour_folder_path):
            submission_json_path = os.path.join(rumour_folder_path, submission_json)
            with open(submission_json_path, "r", encoding='utf-8') as file:
                json_obj = json.load(file)
                sub = json_obj['redditSubmission']
                if sub['IsRumour'] and not sub['IsIrrelevant']:
                    print("Adding {} as rumour".format(submission_json))
                    
                    distinct_comments = dict()
                    rumour_truth = rumour_truth_to_int[sub['TruthStatus']]
                    rumour_count += 1
                    truth_count += rumour_truth
                    for branch in json_obj['branches']:
                        branch_labels = []
                        for comment in branch:

                            label = comment['comment']['SDQC_Submission']
                            created = comment['comment']['created']
                            comment_id = comment['comment']['comment_id']
                            
                            time_stamp = time.mktime(time.strptime(created, "%Y-%m-%dT%H:%M:%S"))
                            distinct_comments[comment_id] = (sdqc_to_int[label], time_stamp)

                            label_distribution[label] += 1
                            branch_labels.append(sdqc_to_int[label])
                    
                    # sort them by time
                    comments_by_time = sorted(distinct_comments.values(), key=lambda x: x[1])
                    
                    if keep_time:
                        # normalize time
                        max_time = max([x[1] for x in comments_by_time])
                        min_time = min([x[1] for x in comments_by_time])
                        
                        for i in range(len(comments_by_time)):
                            if (max_time - min_time) == 0.0:
                                norm_time = 0.0
                            else:
                                norm_time = (comments_by_time[i][1] - min_time) / (max_time - min_time)
                            comments_by_time[i] = (comments_by_time[i][0], norm_time)

                    if keep_time: # unpack tupples to one list
                        label_data.append((rumour_truth, unpack_tupples(comments_by_time)))
                    else:
                        # discard time stamps for now
                        label_data.append((rumour_truth, [x[0] for x in comments_by_time]))
    
    print("Preprocessed {} rumours of which {} were true".format(rumour_count, truth_count))
    print("With sdqc overall distribution: ")
    print(label_distribution)
    return label_data

# read hmm data in top level comment tree structure
def read_hmm_data_cmt_trees(filename, keep_time):
    if not filename:
        print("Cannot run method read_hmm_data_no_branches without filename parameter")
        return
    
    label_data = []

    sdqc_to_int = {'Supporting':0, 'Denying':1, 'Querying':2, 'Commenting':3}
    label_distribution = {'Supporting':0, 'Denying':0, 'Querying':0, 'Commenting':0}
    rumour_count = 0
    truth_count = 0
    comment_trees = dict()
    
    for rumour_folder in os.listdir(filename):
        rumour_folder_path = os.path.join(filename, rumour_folder)
        if not os.path.isdir(rumour_folder_path):
            continue
        for submission_json in os.listdir(rumour_folder_path):
            submission_json_path = os.path.join(rumour_folder_path, submission_json)
            
            with open(submission_json_path, "r", encoding='utf-8') as file:
                json_obj = json.load(file)
                sub = json_obj['redditSubmission']
                if sub['IsRumour'] and not sub['IsIrrelevant']:
                    print("Adding {} as rumour".format(submission_json))
                    
                    distinct_comments = dict()
                    rumour_truth = rumour_truth_to_int[sub['TruthStatus']]
                    rumour_count += 1
                    truth_count += rumour_truth
                    for branch in json_obj['branches']:
                        first_cmt = True
                        branch_top = ''
                        for comment in branch:

                            label = comment['comment']['SDQC_Submission']
                            created = comment['comment']['created']
                            comment_id = comment['comment']['comment_id']
                            time_stamp = time.mktime(time.strptime(created, "%Y-%m-%dT%H:%M:%S"))
                            
                            if first_cmt: # on first comment, init new dict if not there. add top if not there
                                first_cmt = False
                                branch_top = comment_id
                                if comment_id not in comment_trees:
                                    comment_trees[comment_id] = dict()
                                    comment_trees[comment_id][comment_id] = (rumour_truth, sdqc_to_int[label], time_stamp)
                                    label_distribution[label] += 1
                            else:
                                if comment_id not in comment_trees[branch_top]:
                                    comment_trees[branch_top][comment_id] = (rumour_truth, sdqc_to_int[label], time_stamp)
                                    label_distribution[label] += 1
    
    for cmt_top, tree in comment_trees.items():
        comments_by_time = sorted(tree.values(), key=lambda x: x[2])
        
        rumour_truth = comments_by_time[0][0]
        if keep_time:
            # normalize time
            max_time = max([x[2] for x in comments_by_time])
            min_time = min([x[2] for x in comments_by_time])
            
            for i in range(len(comments_by_time)):
                if (max_time - min_time) == 0.0:
                    norm_time = 0.0
                else:
                    norm_time = (comments_by_time[i][2] - min_time) / (max_time - min_time)
                comments_by_time[i] = (comments_by_time[i][0], comments_by_time[i][1], norm_time)
            comments_by_time = unpack_tupples([(x[1], x[2]) for x in comments_by_time])
        else:
            comments_by_time = unpack_tupples([x[1] for x in comments_by_time])
            
        label_data.append((rumour_truth, comments_by_time))
        
                    
    print("Preprocessed {} rumours of which {} were true".format(rumour_count, truth_count))
    print("Found {} comment trees".format(len(comment_trees)))
    print("With sdqc overall distribution: ")
    print(label_distribution)
    return label_data 

def write_hmm_data(filename, data):
    if not data:
        return
    print('Writing hmm vectors to', filename)
    with open(filename, "w+", newline='') as out_file:
        csv_writer = csv.writer(out_file, delimiter='\t')
        csv_writer.writerow(['TruthStatus', 'SDQC_Labels'])
        
        for (truth_status, labels) in data:
            csv_writer.writerow([truth_status, labels])
    print('Done')

def get_hmm_data(filename=hmm_datafile, delimiter=tab):
    data = []
    max_branch_len = 0
    with open(filename) as file:
        has_header = csv.Sniffer().has_header(file.read(1024))
        file.seek(0) # Rewind
        csvreader = csv.reader(file, delimiter=delimiter)
        if has_header:
            next(csvreader) # Skip header row
        for row in csvreader:
            truth_status = int(row[0])
            values = row[1].strip("[").strip("]").split(',')
            instance_vec = [float(i.strip().strip('(').strip(')')) for i in values]
            data.append((truth_status, instance_vec))
            max_branch_len = max(max_branch_len, len(instance_vec))
    
    return data, max_branch_len

def get_semeval_hmm_data(filename=semeval_hmm_data, delimiter=tab):
    data = []
    max_branch_len = 0
    with open(filename) as file:
        has_header = csv.Sniffer().has_header(file.read(1024))
        file.seek(0) # Rewind
        csvreader = csv.reader(file, delimiter=delimiter)
        if has_header:
            next(csvreader) # Skip header row
        for row in csvreader:
            event = row[0]
            truth_status = int(row[1])
            values = row[2].strip("[").strip("]").split(',')
            instance_vec = [float(i.strip()) for i in values]
            data.append((event, truth_status, instance_vec))
            max_branch_len = max(max_branch_len, len(instance_vec))
    
    return data, max_branch_len

    
if __name__ == "__main__":
    main(sys.argv[1:])