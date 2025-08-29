from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
# This version has two issues. 
# 1) Not entire app's components is traced. It only traces component with .invoke function
# 2) Once the all components are loaded and we ask question. then we have to again load all the documets
# again and follow the same pipeline to get output. This makes the process very slow.
# We will solve it in next 4.RAG_version_1.py
os.environ["LANGCHAIN_PROJECT"]="RAG Project"

load_dotenv()
# 1-> Load Documents
loader=PyPDFLoader("docs\\islr.pdf")

print("================Loading Started================")
docs= loader.load()
print("================Docs Loaded================")
# 2-> Split Documents
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=150)
splits=splitter.split_documents(docs)
print("================Docs Splitted================")
# 3-> Create Embeddings of Splited Documents and store in Vector Store
print("================Embedding Creation Started================")
embeddingModel=OpenAIEmbeddings(model="text-embedding-3-small")
vectorStore=FAISS.from_documents(splits,embeddingModel)
print("================Vector Store Created================")
# 4-> creating a retriever
print("================Retriever Created================")
retriever=vectorStore.as_retriever(search_type="similarity",search_kwargs={'k': 5, 'fetch_k': 10})
# 5-> Prompt
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

print("RAG IS READY. Ctrl+C to exit")
question=input("Ask ISLR-GPT: ")
response=chain.invoke(question.strip())
print(response)