import os

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

from utils.web_loader import create_retriever

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = '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'


def rag_url_agent(query: str):
    memory = SqliteSaver.from_conn_string(":memory:")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    tool1 = create_retriever('https://artsandculture.google.com/story/EQURMn3lzjV8JQ',"url")
    tool2 = create_retriever('./documents/pdf/Syllabus.pdf',"pdf")
    tools = [tool1, tool2]

    agent_executor = create_react_agent(llm, tools, checkpointer=memory)

    config = {"configurable": {"thread_id": "abc123"}}

    response = agent_executor.invoke(
        {"messages": [HumanMessage(content=query)]}, config=config
    )
    print(response['messages'])
    print('-------')
    print(StrOutputParser().invoke(response['messages'][-1]))
    # StrOutputParser().batch(res['messages'])[-1]


if __name__ == '__main__':
    # query = "What does television fueled inventors to invent? "
    query = "Who are the instructors and what are their emails? "
    rag_url_agent(query)