# Source: https://korpus.dsl.dk/e-resources/Synonyms%20from%20DDO.html
# Be aware of the following characteristics:
#
# There is one line for each sense of a word which means that for many words there are multiple lines.
# The list may contain multiword expressions, sometimes with parentheses, slashes, dots, etc. for denoting variation.
# As there is one line per sense a synonymic relation between A and B, and A and C, does not imply a synonymic relation
# between A and C, i.e. from ånd = geni and geni = genialitet, you cannot deduce genialitet = ånd. For the same reason
# (one sense per line), you cannot be sure that a synonym for a given word is correct in a given context.
import re


def get_synonyms():
    syn_dict = {}
    vars = re.compile(r'(\s?\([^)]*\)\s?)|(\s?\.\.\s?)|(/\w*)')

    with open('ddo-synonyms.csv', 'r', encoding='utf8') as infile:
        for line in infile:
            # lines are in the format:
            # headword + TAB + a comma-separated list of synonyms
            instance = line.rstrip('\n').lower().split('\t')
            headword = vars.sub('', instance[0])
            synonyms = instance[1].split(';')
            for synonym in synonyms:
                synonym = vars.sub('', synonym)
                if headword in syn_dict:
                    syn_dict[headword].append(synonym)
                else:
                    syn_dict[headword] = [synonym]
    return syn_dict

d = get_synonyms()
with open('synonyms.txt', 'w+', encoding='utf8') as outfile:
    for headword, synonyms in d.items():
        outfile.write('%s: %s\n' % (headword, ', '.join(synonyms)))
