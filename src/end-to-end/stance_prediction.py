import os, json, time, csv, sys
import argparse
from src import data_loader
from src import model_stats
from sklearn.svm import LinearSVC
from sklearn.metrics.classification import classification_report
import numpy as np

preprocessed_folder = '../../data/preprocessed/'
output_folder = '../../output/end-to-end/'

parser = argparse.ArgumentParser(description='Prediction of stance labels')
parser.add_argument('-f', '--file_folder', default='../data/annotated/', help='Input folder holding annotated data')
parser.add_argument('-b', '--branches', default=False, action='store_true', help='Predict and structure in branches')
parser.add_argument('-c', '--conversations', default=False, action='store_true',
                    help='Predict and structure in conversations')
parser.add_argument('-s', '--stats', default=False, action='store_true', help='Write out stats')
args = parser.parse_args()
file_folder = args.file_folder
branch_structure = args.branches
conversation_structure = args.conversations
enable_stats = args.stats

truth_to_id = {
    'False': 0,
    'True': 1,
    'Unverified': 2
}

rand = np.random.RandomState(42)

rumour_submissions = {}
rumour_branches = {}
rumour_conversations = {}

if branch_structure or conversation_structure:
    folder = os.path.join(output_folder, 'branches/')
    with open(os.path.join(folder, 'rumour_branches.txt'), 'r') as infile:
        submissions = infile.read().split('\n\n\n')
        for submission in submissions:
            branches = submission.split('\n\n')
            sub_id = branches[0]
            for branch in branches[1:]:
                branch_post_ids = branch.split('\n')
                if sub_id not in rumour_branches:
                    rumour_branches[sub_id] = []
                rumour_branches[sub_id].append(branch_post_ids)
    if conversation_structure:
        folder = os.path.join(output_folder, 'conversations/')
        for sub_id, branches in rumour_branches.items():
            if sub_id not in rumour_conversations:
                rumour_conversations[sub_id] = {}
            for branch in branches:
                head = branch[0]
                if head not in rumour_conversations[sub_id]:
                    rumour_conversations[sub_id][head] = {head}
                # if not len(branch) > 1:  # Should never happen
                #     continue
                for c_id in branch[1:]:
                    rumour_conversations[sub_id][head].add(c_id)
    output_folder = folder
else:
    output_folder = os.path.join(output_folder, 'submission/')

predicted = []
true = []

with open(os.path.join(preprocessed_folder,
                       'PP_text_sentiment_bow_pos_word2vec300_rumours.csv'), 'r', encoding='utf8') as rumour_file:
    csvreader = csv.reader(rumour_file, delimiter='\t')
    header = next(csvreader)
    for row in csvreader:
        # 'sub_id', 'comment_id', 'time', 'truth', 'sdqc_submission'
        sub_id = row[0]
        c_id = row[1]
        creation_time = row[2]
        time_stamp = time.mktime(time.strptime(creation_time, "%Y-%m-%dT%H:%M:%S"))
        truth = int(row[3])
        sdqc = int(row[4])  # Submission SDQC
        instance_vec = []
        for feature in row[5:]:
            values = feature.strip("[").strip("]").split(',')
            feature_vec = [float(i.strip()) for i in values]
            instance_vec.extend(feature_vec)
        if sub_id not in rumour_submissions:
            rumour_submissions[sub_id] = []
        rumour_submissions[sub_id].append((time_stamp, truth, c_id, sdqc, instance_vec))

    # Map each feature to its corresponding index in the instance vector
    feature_mapping = {}
    for i, feature_name in enumerate(header[5:]):
        feature_mapping[feature_name] = i

X, y, n_features, feature_mapping = \
    data_loader.get_features_and_labels(os.path.join(preprocessed_folder, 'PP_text_sentiment_bow_pos_word2vec300.csv'))
feature_config = data_loader.get_features()
X = data_loader.select_features(X, feature_mapping, feature_config)
X = np.asarray(X, dtype=np.float64, order='C')


def normalize(x_i, min_x, max_x):
    if x_i == 0:
        return 0
    if max_x - min_x != 0:
        return (x_i - min_x) / (max_x - min_x)

    return 0.0


def write_labels(time_to_label, stance_no_time, stance_time, sub_id):
    with open(os.path.join(output_folder, stance_no_time), 'a', newline='') as stance_file, open(
            os.path.join(output_folder, stance_time), 'a', newline='') as stance_file_time:
        stance_writer = csv.writer(stance_file, delimiter='\t')
        stance_time_writer = csv.writer(stance_file_time, delimiter='\t')

        time_to_label.sort(key=lambda tup: tup[0])  # sort by time
        min_time, max_time = time_to_label[0][0], time_to_label[len(time_to_label) - 1][0]

        # Write out stance labels sorted by time, excluding time stamp
        stance_writer.writerow([truth_status, [y for x, y in time_to_label], sub_id])

        # Write out stance labels sorted by time, including normalized time stamps
        y_pred_time_norm = [(y, normalize(t, min_time, max_time)) for t, y in time_to_label]
        stance_time_writer.writerow([truth_status, y_pred_time_norm, sub_id])


def create_files(stance_no_time, stance_time):
    with open(os.path.join(output_folder, stance_no_time), 'w+', newline='') as stance_file, open(
            os.path.join(output_folder, stance_time), 'w+', newline='') as stance_file_time:
        stance_writer = csv.writer(stance_file, delimiter='\t')
        stance_time_writer = csv.writer(stance_file_time, delimiter='\t')
        stance_writer.writerow(['truth_status', 'stance_sequence', 'sub_id'])
        stance_time_writer.writerow(['truth_status', 'stance_sequence', 'sub_id'])


def report_stats(true_labels, predicted_labels, loo_sub_id):
    # Write out stats for the prediction
    cm, acc, f1, sdqc_acc = model_stats.plot_confusion_matrix(true_labels, predicted_labels,
                                                              title='%s confusion matrix' % LOO_sub_id)
    target_names = ['S', 'D', 'Q', 'C']
    cr = classification_report(true_labels, predicted_labels, labels=[0, 1, 2, 3],
                               target_names=target_names, output_dict=True)
    sdqc_f1 = [cr['S']['f1-score'], cr['D']['f1-score'], cr['Q']['f1-score'], cr['C']['f1-score']]
    print('Predict rumour', loo_sub_id)
    print('Acc: %.4f' % acc)
    print('f1: %.4f' % f1)
    print("SDQC acc:", sdqc_acc)
    print('SDQC f1:', sdqc_f1)
    print('True:', true_labels)
    print('Pred:', predicted_labels.tolist())
    print()
    with open(os.path.join(output_folder, LOO_sub_id + '_stance.txt'), 'w+') as rumour_stance_file:
        rumour_stance_file.write(np.array2string(cm) + '\n')
        rumour_stance_file.write('Acc: %.4f\n' % acc)
        rumour_stance_file.write('F1: %.4f\n' % f1)
        rumour_stance_file.write('SDQC acc: {}\n'.format(sdqc_acc))
        rumour_stance_file.write('SDQC f1 : {}\n'.format(sdqc_f1))
        rumour_stance_file.write('True labels:\n')
        rumour_stance_file.write(','.join(str(x) for x in true_labels) + '\n')
        rumour_stance_file.write('Predicted labels:\n')
        rumour_stance_file.write(','.join(str(x) for x in predicted_labels) + '\n')


if branch_structure:
    create_files('stance_labels_branch.csv', 'stance_labels_time_branch.csv')
elif conversation_structure:
    create_files('stance_labels_conv.csv', 'stance_labels_time_conv.csv')
else:
    create_files('stance_labels.csv', 'stance_labels_time.csv')

submission_ids = rumour_submissions.keys()
for LOO_sub_id in submission_ids:  # The submission to leave out
    clf = LinearSVC(penalty='l2', C=10, class_weight=None, dual=True, max_iter=50000, random_state=rand)
    rumour_test = None  # The rumour to test and predict stance labels for
    rumour_train = []  # The rest of the rumour data, used for training
    # Go through the rumour train data and append it to the non-rumour train data
    for sub_id, values in rumour_submissions.items():
        if sub_id == LOO_sub_id:
            rumour_test = values
            continue
        rumour_train.extend(values)
    truth_status = rumour_test[0][1]  # Store truth label
    rumour_X_train = [x[4] for x in rumour_train]
    rumour_X_train = np.asarray(rumour_X_train, dtype=np.float64, order='C')
    rumour_y_train = [x[3] for x in rumour_train]
    X_all = np.append(X, rumour_X_train, axis=0)
    y_all = []
    y_all.extend(y)
    y_all.extend(rumour_y_train)

    # Train and make the prediction
    clf.fit(X_all, y_all)
    rumour_X_test = [x[4] for x in rumour_test]
    rumour_X_test = np.asarray(rumour_X_test, dtype=np.float64, order='C')
    rumour_y_test = [x[3] for x in rumour_test]
    y_true, y_pred = rumour_y_test, clf.predict(rumour_X_test)
    true.extend(y_true)
    predicted.extend(y_pred)
    id_to_y_pred = dict([(x[2], (x[0], y)) for x, y in zip(rumour_test, y_pred)])

    if branch_structure:
        branches = rumour_branches[LOO_sub_id]
        for branch in branches:
            branch_time_label = [id_to_y_pred[c_id] for c_id in branch]
            write_labels(branch_time_label, 'stance_labels_branch.csv', 'stance_labels_time_branch.csv', LOO_sub_id)
    elif conversation_structure:
        conversations = rumour_conversations[LOO_sub_id]
        for conversation in conversations.values():
            conversations_time_label = [id_to_y_pred[c_id] for c_id in conversation]
            write_labels(conversations_time_label, 'stance_labels_conv.csv', 'stance_labels_time_conv.csv', LOO_sub_id)
    else:
        y_pred_time = [(x[0], y) for x, y in zip(rumour_test, y_pred)]
        write_labels(y_pred_time, 'stance_labels.csv', 'stance_labels_time.csv', LOO_sub_id)
        y_true_time = [(x[0], y) for x, y in zip(rumour_test, y_true)]
        write_labels(y_pred_time, 'stance_labels_true.csv', 'stance_labels_time_true.csv', LOO_sub_id)

    if enable_stats:
        report_stats(y_true, y_pred, LOO_sub_id)

report_stats(true, np.asarray(predicted), 'all')
model_stats.plot_confusion_matrix(true, predicted, title='Automatic rumour stance labels',
                                  save_to_filename=os.path.join(output_folder, 'cm.png'))
