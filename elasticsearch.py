import os
from subprocess import Popen, PIPE, STDOUT

from haystack.document_stores import ElasticsearchDocumentStore

import requests

if __name__ == "__main__":
    # print("Hello World")
    document_store = ElasticsearchDocumentStore()
