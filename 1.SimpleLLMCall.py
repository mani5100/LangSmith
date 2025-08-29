from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

prompt=PromptTemplate(
    template="""Answer this question. {question}""",
    input_variables=['question']
)
model=ChatOpenAI()
parser=StrOutputParser()

chain=prompt|model|parser
response=chain.invoke({
    "question":"What is capital of Pakistan?"
})
print(response)