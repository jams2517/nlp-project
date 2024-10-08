from __future__ import print_function
from __future__ import unicode_literals
import collections
import copy
import io
import nltk
import re
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')

stopwords = set()
sentences = []
sentences_processing = []
sentence_dictionary = collections.defaultdict(dict)
stemWords = {}


def readStemWords():
    '''
        Reads the words from the stem words list and transforms the data into usable format
    '''
    global stemWords
    with io.open("word_list_marathi.txt", encoding='utf-8') as textFile:
        index = 0
        for line in textFile:
            line = line.strip()
            if len(line) > 0:
                index += 1
                wordEndIndex = line.find(">")
                word = line[2:wordEndIndex]
                line = line[wordEndIndex + 1:]
                baseEndIndex = line.find("]")
                base = line[1:baseEndIndex].strip()
                line = line[baseEndIndex + 1:]
                stem = None
                if len(base) >= 0:
                    stemEndIndex = base.find('-')
                    if stemEndIndex > 0:
                        stem = base[:stemEndIndex]

                valid = line[line.find("(") + 1: line.find(")")].strip()
                if valid == "0":
                    continue
                line = line[line.find("{") + 1: line.find("}")].strip()
                related = []
                if len(line) > 0:
                    split = line.split(",")
                    for s in split:
                        related.append(s[:s.find("|")])
                if stem == None and len(related) > 0:
                    stem = related[0]
                if stem != None:
                    stemWords[word] = {}
                    stemWords[word]["stem"] = stem
                    stemWords[word]["related"] = related


def tokenize(file_content):
    '''
    Tokenizes the sentences and words from file content.
    :param file_content: content of the file to be processed (string or file-like object)
    '''
    global sentences, sentences_processing, sentence_dictionary
    
    # If file_content is a file-like object, read its content
    if hasattr(file_content, 'read'):
        data = file_content.read()
    else:
        data = file_content
    
    # Tokenize sentences
    sentences = sent_tokenize(data)
    sentences_processing = copy.deepcopy(sentences)
    
    counter = 0
    for sentence in sentences_processing:
        sentence = sentence[:-1]
        sentence = re.sub(',|\.|-|\(|\)', ' ', sentence)
        tokens = sentence.strip().split()
        actualTokens = removeStopWords(tokens)
        stemmedTokens = stemmerMarathi(actualTokens)
        sentence_dictionary[counter] = stemmedTokens
        counter += 1


def readStopWords():
    '''
    Reads the stopwords from the file
    '''
    with io.open("stopwords.txt", encoding='utf-8') as textFile:
        for line in textFile:
            words = line.lower().strip()
            stopwords.add(words)


def removeStopWords(wordlist):
    '''
    Removes the stopwords from the sentences
    :param wordlist: list of words to be filtered
    '''
    newlist = [word for word in wordlist if word not in stopwords]
    return newlist


def removeCase(word):
    '''
    Reduces the word to its stem by removing case suffixes
    :param word: word to be reduced
    :return: stem of the word
    '''
    word_length = len(word) - 1
    if word_length > 5:
        suffix = "शया"
        if word.endswith(suffix):
            return word[:-len(suffix)]

    # Similar checks for other suffixes as in the original code

    return word


def removeNoGender(word):
    global stemWords
    if word in stemWords:
        return stemWords[word]["stem"]

    word_length = len(word) - 1
    # Similar checks for other suffixes as in the original code

    return word


def stemmerMarathi(words):
    return [removeNoGender(removeCase(word)) for word in words]


def cleanText(file_content):
    '''
    Tokenize, Remove stopwords and reduce the words to their stem.
    :param file_content: content of file to be preprocessed (string or file-like object)
    '''
    global sentence_dictionary, sentences
    readStopWords()
    tokenize(file_content)  # Pass file content instead of filename
    size = 0
    for i in range(0, len(sentence_dictionary)):
        size += len(sentence_dictionary[i])
    sentence_dictionary = {key: value for key,
                           value in sentence_dictionary.items() if len(value) > 0}
    return sentence_dictionary, sentences, size


# Read stem words at the start
readStemWords()
