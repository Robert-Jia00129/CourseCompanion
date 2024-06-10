import os

from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .utils import get_website_name

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = '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'


def create_retriever(path: str, document_type: str):
    if document_type.lower() == "pdf":
        loader = PyPDFLoader(path, extract_images=True)
        file_name = os.path.basename(path)
        tool_name = f"retriever_for_{file_name}"
        tool_description = f"queries the pdf `{file_name}` for relevant information"

    elif document_type.lower() == "url":
        loader = WebBaseLoader(
                web_paths=(path,),
                )
        website_name = get_website_name(path)
        tool_name = f"retriever_for_{website_name}"
        tool_description = f"queries the website `{website_name}` for relevant information"

    else:
        raise NotImplementedError

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0) # make sure <= 1024 tokens, could find alternatives
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6})

    tool = create_retriever_tool(
        retriever,
        tool_name,
        tool_description,
    )
    return tool

#
#
# def rag_url_custom_prompt(query: str):
#     # setup
#     llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
#
#     ### Construct retriever ###
#     docs = load_from_url('https://artsandculture.google.com/story/EQURMn3lzjV8JQ')
#
#     print(f'Loaded content from website: {docs[0].page_content[:500]}...')
#     # TODO: handle the case where docs are too long
#
#
#     # store
#     vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
#     # retrieve and generate
#     retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
#     # retrieved_docs = retriever.invoke(question)
#     # len(retrieved_docs)
#
#     system_prompt = (
#         "You are an assistant for question-answering tasks. "
#         "Use the following pieces of retrieved context to answer "
#         "the question. If you don't know the answer, say that you "
#         "don't know. Use three sentences maximum and keep the "
#         "answer concise."
#         "\n\n"
#         "{context}"
#     )
#
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system_prompt),
#             ("human", "{input}"),
#         ]
#     )
#     question_answer_chain = create_stuff_documents_chain(llm,prompt)
#     rag_chain = create_retrieval_chain(retriever, question_answer_chain)
#
#     response = rag_chain.invoke({"input": query})
#     print(response["answer"])

if __name__ == '__main__':
    pass


