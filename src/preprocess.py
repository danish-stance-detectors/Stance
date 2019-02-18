import os
from gensim.test.utils import datapath

datafolder = '../data/'
fasttext_data = os.path.join(datafolder, 'cc.da.300.bin')

cap_path = datapath(fasttext_data)
fb_partial = FastText.load_fasttext_format(cap_path, encoding='utf-8', full_model=False)
'æble' in fb_partial.wv.vocab