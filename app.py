from flask import Flask, request, render_template
from preprocess import cleanText
import networkx
import itertools
import math
import io

# Initialize Flask app
app = Flask(__name__)

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
        graph.add_edge(edge[0], edge[1],
                       weight=getSimilarity(edge[0], edge[1]))
    return graph

# TextRank Similarity function modified to handle file content directly
def textRankSimilarity(fileContent):
    global sentenceDictionary
    summarySentenceCount = 5
    sentenceDictionary = {}
    sentences = []
    # Pass the file content (string) to the cleanText function
    sentenceDictionary, sentences, size = cleanText(io.StringIO(fileContent))
    graph = generateGraph(list(sentenceDictionary.keys()))
    pageRank = networkx.pagerank(graph)
    output = "\n".join([sentences[sentenceID] for sentenceID in sorted(
        sorted(pageRank, key=pageRank.get, reverse=True)[:summarySentenceCount])])
    return output

# Flask route for the summarization form
@app.route("/", methods=["GET", "POST"])
def summarize():
    if request.method == "POST":
        # Check if a file was uploaded
        uploaded_file = request.files.get("file")
        if uploaded_file:
            # Read the file content
            file_content = uploaded_file.read().decode("utf-8")
            # Call the textRankSimilarity function with the file content
            summary = textRankSimilarity(file_content)
            return render_template("index.html", original_text=file_content, summary=summary)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
