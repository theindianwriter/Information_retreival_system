from mediawiki import MediaWiki
wikipedia = MediaWiki()
import re
import json
import random

docs_json = json.load(open("cranfield/cran_docs.json", 'r'))[:]

docs_title = [item["title"] for item in docs_json]
docs_id = [item["id"] for item in docs_json]
wiki_docs = []
id = 0

for title,doc_id in zip(docs_title,docs_id):
    if title  == "":
        continue
    title = re.sub(r'[^\w\s]', '',title)
    if title == "":
        continue
    print(doc_id)
    wiki_article_titles = wikipedia.search(title)
    for wiki_article_title in wiki_article_titles:
        try:
            p = wikipedia.page(wiki_article_title)
        except:
            continue
            
        wiki_docs.append({"id" : id,"content" : p.content})
        

        with open('WikipediaAnother/'+str(id)+".txt",'w') as file:
            file.writelines(p.summary)

        id += 1



    

    

