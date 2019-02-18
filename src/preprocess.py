import os
from gensim.models.fasttext import FastText

datafolder = '../data/'
fasttext_data = os.path.join(datafolder, 'fasttext/cc.da.300.bin')

fb = FastText.load_fasttext_format(fasttext_data, encoding='utf8')