
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import requests
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
import os
os.environ["LANGCHAIN_PROJECT"]="Agent"
load_dotenv()
@tool
def get_weather_data(city:str):
    "This is tool to get Weather forecast Data"
    url=f"https://api.weatherstack.com/current?access_key=b0f89309de912775e2238ca2bedc411c&query={city}"
    response=requests.get(url)
    return response.json()
search_tool = TavilySearch(tavily_api_key=os.environ['TRAVITY_API_KEY'])
prompt=hub.pull("hwchase17/react")
agent=create_react_agent(
    llm=ChatOpenAI(),
    tools=[search_tool,get_weather_data],
    prompt=prompt
)
agent_executpr=AgentExecutor(
    agent=agent,
    verbose=True,
    tools=[search_tool,get_weather_data]
)
config={
    "run_name":"Agent"
}
response=agent_executpr.invoke({"input":"what is city of pakistan that is famous for oranges and what is current weather condition"},config=config)
print(response['output'])