from __future__ import print_function
import collections
import io
import math
import operator
import sys
import networkx as nx
from preprocess import cleanText

window = 10
numberofSentences = 6
nodeHash = {}
textRank = {}
sentenceDictionary = collections.defaultdict(dict)
size = 0
sentences = []

def generatepositionaldistribution():
    global nodeHash, sentenceDictionary, sentences, size
    positional_dictionary = collections.defaultdict(dict)
    count = 0
    for i in sentenceDictionary.keys():
        for j in range(0, len(sentenceDictionary[i])):
            count += 1
            position = float(count) / (float(size) + 1.0)
            positional_dictionary[i][j] = 1.0 / (math.pi * math.sqrt(position * (1 - position)))
            word = sentenceDictionary[i][j]
            if word in nodeHash:
                if nodeHash[word] < positional_dictionary[i][j]:
                    nodeHash[word] = positional_dictionary[i][j]
            else:
                nodeHash[word] = positional_dictionary[i][j]

def textrank():
    global sentenceDictionary, nodeHash, textRank
    graph = nx.Graph()
    graph.add_nodes_from(nodeHash.keys())
    for i in sentenceDictionary.keys():
        for j in range(0, len(sentenceDictionary[i])):
            current_word = sentenceDictionary[i][j]
            next_words = sentenceDictionary[i][j + 1:j + window]
            for word in next_words:
                graph.add_edge(current_word, word, weight=(nodeHash[current_word] + nodeHash[word]) / 2)
    textRank = nx.pagerank(graph, weight='weight')
    keyphrases = sorted(textRank, key=textRank.get, reverse=True)[:n]
    return keyphrases

def summarize(filePath, keyphrases, numberofSentences):
    global textRank, sentenceDictionary, sentences
    sentenceScore = {}
    for i in sentenceDictionary.keys():
        position = float(i + 1) / (float(len(sentences)) + 1.0)
        positionalFeatureWeight = 1.0 / (math.pi * math.sqrt(position * (1.0 - position)))
        sumKeyPhrases = 0.0
        for keyphrase in keyphrases:
            if keyphrase in sentenceDictionary[i]:
                sumKeyPhrases += textRank[keyphrase]
        sentenceScore[i] = sumKeyPhrases * positionalFeatureWeight
    sortedSentenceScores = sorted(sentenceScore.items(), key=operator.itemgetter(1), reverse=True)[:numberofSentences]
    sortedSentenceScores = sorted(sortedSentenceScores, key=operator.itemgetter(0), reverse=False)
    summary = []
    for i in range(0, len(sortedSentenceScores)):
        summary.append(sentences[sortedSentenceScores[i][0]])
    return "".join(summary)

def process(arg1):
    arg2 = 5
    arg3 = 6
    global window, n, numberofSentences, textRank, sentenceDictionary, size, sentences
    if arg1 != None and arg2 != None and arg3 != None:
        sentenceDictionary, sentences, size = cleanText(io.StringIO(arg1))
        window = int(arg3)
        numberofSentences = int(arg2)
        n = int(math.ceil(min(0.1 * size, 7 * math.log(size))))
        generatepositionaldistribution()
        keyphrases = textrank()
        return summarize(arg1, keyphrases, numberofSentences)
    else:
        return "Not enough parameters."

def summarize_positional(input_text):
    """
    Wrapper function to use the process method for positional summarization.
    :param input_text: The input text to summarize
    :return: The summarized text
    """
    return process(input_text)

if __name__ == "__main__":
    # No need for streamlit or UI code in this version
    print("This script is ready to be used as a module.")
