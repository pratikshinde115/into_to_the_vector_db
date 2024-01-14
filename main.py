
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from langchain.vectorstores import Pinecone
import pinecone
from langchain.chains import RetrievalQA


from langchain_openai import OpenAI


pinecone.init(api_key = "" , environment = "gcp-starter")
 
if __name__ == "__main__":
        loaders = TextLoader('/home/pratik/Desktop/intro_to_vector_db/mediumblogs/mediumblog1.txt')


        doc = loaders.load()
        print(doc)
        text_splitter = CharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 0)
        texts = text_splitter.split_documents(doc)
        print(len(texts))

        Embeddings = OpenAIEmbeddings(openai_api_key = os.environ.get("OPENAI_API_KEY"))
        docsearch = Pinecone.from_documents(texts , Embeddings , index_name = 'medium-blog-embeddings-index')


        qa = RetrievalQA.from_chain_type(
                llm=OpenAI() , chain_type = "stuff" , retriever = docsearch.as_retriever()
        )
        querry = "what is similarity measures"
        result = qa({'query' : querry})
        print(result)
