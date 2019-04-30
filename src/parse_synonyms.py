# Source: https://korpus.dsl.dk/e-resources/Synonyms%20from%20DDO.html
# Be aware of the following characteristics:
#
# There is one line for each sense of a word which means that for many words there are multiple lines.
# The list may contain multiword expressions, sometimes with parentheses, slashes, dots, etc. for denoting variation.
# As there is one line per sense a synonymic relation between A and B, and A and C, does not imply a synonymic relation
# between A and C, i.e. from ånd = geni and geni = genialitet, you cannot deduce genialitet = ånd. For the same reason
# (one sense per line), you cannot be sure that a synonym for a given word is correct in a given context.
import re

all_vars = re.compile(r'(\s?\([^)]*\)\s?)|(\s?\.\.\s?|(/\w*))')  # (..) or '..' or '/'
#vars = re.compile(r'(\s?\([^)]*\)\s?)|(\s?\.\.\s?)')  # (..) or '..'
dots = re.compile(r'(\s?\.\.\s?)|(,\s)')

def find_paren_alternatives(line):
    versions = []
    if '(' and ')' in line:
        lpar = line.index('(')
        rpar = line.index(')')
        left = line[:lpar].strip()
        right = line[rpar+1:].strip()
        body = line[lpar+1:rpar].strip()
        body = dots.sub('', body).rstrip(',').strip()
        versions.append(('%s %s' % (left, right)).strip())
        versions.append(('%s %s %s' % (left, body, right)).strip())
    final_versions = []
    for version in versions:
        if '(' and ')' in version:
            new_versions = find_paren_alternatives(version)
            final_versions.extend(new_versions)
        else:
            final_versions.append(version)
    return final_versions

def find_slash_alternatives(line):  # TODO: Hande multiple in one sentence
    tokens = line.split()
    alternatives = []
    pos = 0
    for i, token in enumerate(tokens):
        if '/' in token:
            pos = i
            options = token.split('/')
            for option in options:
                alternatives.append(option)
            break  # assume only one word in a sentence to have alternatives
    lines = []
    if alternatives:
        left = ' '.join(tokens[:pos])
        right = ' '.join(tokens[pos+1:])
        for alternative in alternatives:
            lines.append('%s %s %s' % (left, alternative, right))
    return lines

def find_alternatives(line):
    if '(' and ')' in line:
        versions = find_paren_alternatives(line)
        for version in versions:
            if '/' in version:
                return find_slash_alternatives(version)
        return versions
    if '/' in line:
        return find_slash_alternatives(line)
    return [line]  # none found

def load_synonyms():
    syn_dict = {}
    with open('../data/corpus/ddo-synonyms.csv', 'r', encoding='utf8') as infile:
        for line in infile:
            # lines are in the format:
            # headword + TAB + a comma-separated list of synonyms
            instance = line.rstrip('\n').lower().split('\t')
            headwords = find_alternatives(dots.sub(' ', instance[0]).strip())
            synonyms = instance[1].split(';')
            for headword in headwords:
                for synonym in synonyms:
                    synonym = all_vars.sub('', synonym)
                    if headword in syn_dict:
                        syn_dict[headword].append(synonym)
                    else:
                        syn_dict[headword] = [synonym]
    return syn_dict


with open('synonyms.txt', 'w+', encoding='utf8') as outfile:
    for h, s in load_synonyms().items():
        outfile.write('%s:%s\n' % (h, ';'.join(s)))
