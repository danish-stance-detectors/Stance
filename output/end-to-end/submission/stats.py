import os, ast
for file in os.listdir('.'):
    if file == 'all_stance.txt':
        sdqc_distr = {
            0: 0,
            1: 0,
            2: 0,
            3: 0
        }
        with open(file, 'r') as stat_file:
            lines = stat_file.readlines()
            index = 11
            for i, line in enumerate(lines):
                if 'True' in line:
                    index = i+1
            true_labels = ast.literal_eval(lines[index])
            for y in true_labels:
                sdqc_distr[y] += 1
        with open(file.rstrip('.txt') + '_sdqc.txt', 'w+') as sdqc_file:
            for sdqc, count in sdqc_distr.items():
                sdqc_file.write('{}\t{}\n'.format(sdqc, count))
