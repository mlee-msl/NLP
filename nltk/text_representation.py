import numpy as np
import nltk.tokenize as tokenize
from nltk.corpus import stopwords
import nltk.tag as tag
import nltk.stem.wordnet as wordnet
import nltk.probability as prob
import nltk.text as text


def load_romance():
    corpus = ['When a cigarette falls in love with a match, it is destined to be hurt.',
              'Love is like a butterfly. It goes where it pleases and it pleases where it goes.',
              'If I had a single flower for every time I think about you, I could walk forever in my garden.',
              'Look into my eyes you will see what you mean to me.',
              'If I know what love is, it is because of you.',
              'No one indebted for others,while many people don\'t know how to cherish others.',
              'No matter the ending is perfect or not, you cannot disappear from my world.',
              'Where there is great love, there are always miracles.']
    return corpus


def preprocessing(document):
    # eliminated stopwords
    filtered_words = [word for word in tokenize.word_tokenize(document) if word not in stopwords.words('english')]
    word_tags = [(word, tag_convert(pos)) for word, pos in tag.pos_tag(filtered_words)]
    lemma = wordnet.WordNetLemmatizer()
    # lemmatization
    words = [lemma.lemmatize(word, pos=tag).lower() for word, tag in word_tags]
    return words


def tag_convert(tag):
    if tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        tag = 'v'
    else:
        tag = 'n'
    return tag


def get_lexicons(n_words):
    """
        return lexicons from corpus who has been tokenized
    """
    lexicons = []
    for words in n_words:
        lexicons.extend(words)
    return list(set(lexicons))


def sow(words, lexicons):  # Set of Words
    word_vec = [0]*len(lexicons)
    for i, lexicon in enumerate(lexicons):
        if lexicon in words:
            word_vec[i] = 1  # whether occur or not, if do, then 1 else 0
    return word_vec


def bow(words, lexicons):  # Bag of Words
    word_vec = [0]*len(lexicons)
    fdist = prob.FreqDist()
    for word in words:
        fdist[word.lower()] += 1
    fdist = dict(fdist.most_common(50))  # the 50 most common elements represent the text
    for i, lexicon in enumerate(lexicons):
        if lexicon in words:
            word_vec[i] = fdist[lexicon]  # whether occur or not, if do, then frequency, else 0
    return word_vec


def tf_idf(words, lexicons, corpus):  # Term-Frequency_Inverse-Document-Frequency
    corpus_ = []
    for document in corpus:
        corpus_.append(' '.join(preprocessing(document)))  # preprocessing corpus
    dealed_corpus = text.TextCollection(corpus_)
    word_vec = [0]*len(lexicons)
    for i, lexicon in enumerate(lexicons):
        # calculate tf_idf value for each lexicon in the lexicons, if lexicon not in sentence, then obviously 0, else tf_idf value
        word_vec[i] = dealed_corpus.tf_idf(lexicon, ' '.join(words))
    return word_vec


if __name__ == '__main__':
    corpus = load_romance()
    n_words = []
    for document in corpus:
        n_words.append(preprocessing(document))
    lexicons = get_lexicons(n_words)
    sow_representation = []
    bow_representation = []
    tf_idf_representation = []
    for document in corpus:
        sow_representation.append(sow(preprocessing(document), lexicons))
        bow_representation.append(bow(preprocessing(document), lexicons))
        tf_idf_representation.append(tf_idf(preprocessing(document), lexicons, corpus))
    print(np.array(sow_representation), np.array(bow_representation), np.array(tf_idf_representation), sep='\n')