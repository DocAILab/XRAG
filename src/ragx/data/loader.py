import os
import json
from llama_index.core import Document
from config import Config

cfg = Config()


def get_documents(sources):
    # dirs = os.listdir(path)
    # documents = []
    # for dir in dirs:
    #     files = os.listdir(os.path.join(path,dir))
    #     for file in files:
    #         # read json file
    #         with open(os.path.join(path,dir,file),'r',encoding='utf-8') as f:
    #             for line in f.readlines():
    #                 raw = json.loads(line)
    #                 ducument = Document(text=raw['text'],metadata={'title':raw['title']},doc_id=raw['id'])
    #                 documents.append(ducument)
    # return documents
    title2sentenses = sources['title2sentences']
    title2id = sources['title2id']
    documents = [Document(text=' '.join(sentence_list), metadata={'title': title, 'id': title2id[title]},
                          doc_id=str(title2id[title])) for title, sentence_list in title2sentenses.items()]
    if cfg.experiment_1:
        documents = documents[:cfg.test_all_number_documents]
    return documents


if __name__ == '__main__':
    documents = get_documents('../wiki')
    print(documents)
