import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"]='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'

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
question = "What does television fueled inventors to invent? "

# Fetch and load
loader = WebBaseLoader(
    web_paths=('https://artsandculture.google.com/story/EQURMn3lzjV8JQ',),
)
text = loader.load()
print(f'Successfully loaded content from website: {text[0].page_content[:500]}...')


# store
vectorstore = Chroma.from_documents(documents=text, embedding=OpenAIEmbeddings())
# retrieve and generate
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
# retrieved_docs = retriever.invoke(question)
# len(retrieved_docs)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm,prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


response = rag_chain.invoke({"input": question})
print(response["answer"])
