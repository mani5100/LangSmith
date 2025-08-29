from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
import os
# This version has two issues. 
# 1) Not entire app's components is traced. It only traces component with .invoke function
# 2) Once the all components are loaded and we ask question. then we have to again load all the documets
# again and follow the same pipeline to get output. This makes the process very slow.
# We will solve it in next 4.RAG_version_1.py
os.environ["LANGCHAIN_PROJECT"]="RAG Project With Traceable"

load_dotenv()
# 1-> Load Documents
@traceable(name="Loader")
def load_doc(path:str):
    print("================Loading Started================")
    loader=PyMuPDFLoader(path)
    docs= loader.load()
    print("================Docs Loaded================")
    return docs

@traceable(name="Splitter")
def split_docs(docs,chunk_size,chunk_overlap):
    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    splits=splitter.split_documents(docs)
    print("================Docs Splitted================")
    return splits

@traceable(name="VectorStore") 
def build_vectorstore(splits):
    print("================Embedding Creation Started================")
    embeddingModel=OpenAIEmbeddings(model="text-embedding-3-small")
    vectorStore=FAISS.from_documents(splits,embeddingModel)
    print("================Vector Store Created================")
    return vectorStore

@traceable(name="PipeLine Setup")
def pipeline(path:str):
    docs=load_doc(path)
    splits=split_docs(docs,1000,150)
    vs=build_vectorstore(splits)
    return vs

@traceable(name="Full RAG PipeLine")
def full_rag_pipiline(path:str,question:str):
    vectorStore=pipeline(path)
    retriever=vectorStore.as_retriever(search_type="similarity",search_kwargs={'k': 5, 'fetch_k': 10})


    prompt=ChatPromptTemplate([
        ("system","Answer only from the given context. If you don't find context relevent say sorry i don't know."),
        ("human","Question: {Question} \n\n Context: {Context}")
    ])
    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
    parser=StrOutputParser()
    def formatdocs(docs):
        return "\n\n".join(d.page_content for d in docs)


    parallel_chain=RunnableParallel({
        'Question':RunnablePassthrough(),
        'Context':retriever|RunnableLambda(formatdocs)
    })
    chain=parallel_chain|prompt|llm|parser
    print("================Chains Created================")
    return chain.invoke(question.strip())

if __name__ == "__main__":
    print("RAG IS READY. Ctrl+C to exit")
    question=input("Ask ISLR-GPT: ")
    response=full_rag_pipiline("docs\\islr.pdf",question)
    # config={
    #     "run_name":"RAG_Using_Traceable"
    # }
    print(response)