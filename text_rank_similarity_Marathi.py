import streamlit as st
from preprocess import cleanText
import networkx
import itertools
import math
import io


# Global dictionary to store sentence information
sentenceDictionary = {}

# Function to compute similarity between two sentences
def getSimilarity(sentenceID1, sentenceID2):
    commonWordCount = len(set(sentenceDictionary[sentenceID1]) & set(
        sentenceDictionary[sentenceID2]))
    denominator = math.log(len(
        sentenceDictionary[sentenceID1])) + math.log(len(sentenceDictionary[sentenceID2]))
    return commonWordCount / denominator if denominator else 0

# Function to generate a similarity graph
def generateGraph(nodeList):
    graph = networkx.Graph()
    graph.add_nodes_from(nodeList)
    edgeList = list(itertools.product(nodeList, repeat=2))
    for edge in edgeList:
        graph.add_edge(edge[0], edge[1], weight=getSimilarity(edge[0], edge[1]))
    return graph

# Text Rank Similarity function to handle file content directly
def textRankSimilarity(fileContent):
    global sentenceDictionary
    summarySentenceCount = 5  # You can adjust the number of summary sentences here
    sentenceDictionary = {}
    sentences = []
    
    # Pass the file content (string) to the cleanText function
    sentenceDictionary, sentences, size = cleanText(io.StringIO(fileContent))

    # Generate the similarity graph
    graph = generateGraph(list(sentenceDictionary.keys()))
    pageRank = networkx.pagerank(graph)

    # Safeguard for accessing sentences by sentenceID
    valid_sentence_ids = [sentenceID for sentenceID in sorted(pageRank, key=pageRank.get, reverse=True)
                          if sentenceID < len(sentences)]

    # Limit the summary to summarySentenceCount
    valid_sentence_ids = valid_sentence_ids[:summarySentenceCount]

    # Return the summarized text
    output = "\n".join([sentences[sentenceID] for sentenceID in valid_sentence_ids])
    return output


if __name__ == "__main__":
    # Streamlit UI components
    st.markdown("<h1 style='text-align: center;'>Text Summarization</h1>", unsafe_allow_html=True)
    
    # File uploader
    uploaded_files = st.file_uploader('Upload text file', type=['txt'], accept_multiple_files=False)
    
    if uploaded_files is not None:
        # Read the uploaded file and decode
