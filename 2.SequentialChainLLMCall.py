from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os
load_dotenv()
os.environ["LANGCHAIN_PROJECT"]="sequential-chain"

prompt1=PromptTemplate(
    template="""Generate a detailed report on this user topic.
    Topic: {topic}""",
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template="""You will be given a detailed report and you have to create a 5 point summary of that report.
    Reportt: {report}""",
    input_variables=['report']
)
model1=ChatOpenAI(model="gpt-4o-mini",temperature=0.7)
model2=ChatOpenAI(model="gpt-4o",temperature=0.5)
parser=StrOutputParser()

chain=prompt1|model1|parser|prompt2|model2|parser
config={
    "run_name":"Sequential Chain",
    "tags":["LLM app","Report Generation"],
    "metadata":{
        "model1":{
            "model":"gpt-4o-mini",
            "temprature":0.7
        },
        "model2":{
            "model":"gpt-4o",
            "temprature":0.5
        },
        "parser":"StrOutputParser"
    }
}
response=chain.invoke({
    "topic":"Tourism of Pakistan"},config=config)
print(response)