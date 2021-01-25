from flask import Flask, request, Response
from haystack import Finder
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts
from haystack.reader.farm import FARMReader
from haystack.utils import print_answers
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.retriever.sparse import TfidfRetriever
import json

app = Flask(__name__)

class Haystack_HSBC:
  def __init__(self, data_path, model_path):
    self.document_store = InMemoryDocumentStore()    
    dicts = convert_files_to_dicts(dir_path=data_path, clean_func=clean_wiki_text, split_paragraphs=True)
    self.document_store.write_documents(dicts)
    self.retriever_tfid = TfidfRetriever(document_store=self.document_store)
    self.reader = FARMReader(model_name_or_path=model_path)
    self.finder = Finder(self.reader, self.retriever_tfid)
  def answer_finder(self, question, retrieved_document, no_of_answers):    
    haystack_answer = self.finder.get_answers(question=question, top_k_retriever=int(retrieved_document), top_k_reader=int(no_of_answers))
    #print_answers(prediction, details="medium")
    return haystack_answer

haystack_obj = Haystack_HSBC("data", "my_model")

@app.route("/answers", methods=["GET"])
def answers():
    if request.method == "GET":
        if request.json.get('question'):
            question = request.json.get('question')
        else:
            return Response(response=json.dumps("Error: No question provided."), status=401, mimetype="application/json")
        if request.json.get('retreiver_number'):
            retreiver_number = request.json.get('retreiver_number')
        else:
            return Response(response=json.dumps("Error: No question provided."), status=401, mimetype="application/json")
        if request.json.get('answer_number'):
            answer_number = request.json.get('answer_number')
        
        out = haystack_obj.answer_finder(question, retreiver_number, answer_number)
        return Response(response=json.dumps(out), status=200, mimetype="application/json")


if __name__ == "__main__":
    app.run(debug=True)