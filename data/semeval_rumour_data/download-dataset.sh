#/bin/bash
wget https://s3-eu-west-1.amazonaws.com/downloads.gate.ac.uk/pheme/semeval2017-task8-dataset.tar.bz2
wget http://alt.qcri.org/semeval2017/task8/data/uploads/rumoureval2017-test.tar.bz2
tar -xf semeval2017-task8-dataset.tar.bz2
tar -xf rumoureval2017-test.tar.bz2
rm semeval2017-task8-dataset.tar.bz2
rm rumoureval2017-test.tar.bz2