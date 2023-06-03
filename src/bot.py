import gradio as gr
from langchain.embeddings.llamacpp import LlamaCppEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

persist_directory = "db"
embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-base", model_kwargs={"device": "cuda"}
)
db = Chroma(persist_directory="db", embedding_function=embeddings)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)
callbacks = [StreamingStdOutCallbackHandler()]
model = GPT4All(
    model="../model/ggml-gpt4all-j-v1.3-groovy.bin",
    backend="gptj",
    callbacks=callbacks,
    n_threads=9,
    verbose=True,
)

# qa = RetrievalQA.from_chain_type(
#     llm=model,
#     chain_type="stuff",
#     retriever=db.as_retriever(),
#     return_source_documents=True,
# )

# while True:
#     query = input("\nEnter a query: ")
#     if query == "exit":
#         break
#     # Get the answer from the chain
#     res = qa(query)
#     answer, docs = res["result"], res["source_documents"]
#
#     # Print the result
#     print("\n\n> Question:")
#     print(query)
#     print("\n> Answer:")
#     print(answer)
#
#     # # Print the relevant sources used for the answer
#     print(
#         "----------------------------------SOURCE DOCUMENTS---------------------------"
#     )
#     for document in docs:
#         print("\n> " + document.metadata["source"] + ":")
#         print(document.page_content)
#     print(
#         "----------------------------------SOURCE DOCUMENTS---------------------------"
#     )

qa = ConversationalRetrievalChain.from_llm(
    llm=model,
    chain_type="stuff",
    retriever=db.as_retriever(),
    memory=memory,
    get_chat_history=lambda x: x,
    verbose=True,
)
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def respond(message, chat_history=[]):
        res = qa(
            {
                "question": message,
                "chat_history": chat_history,
                "vectordbkwargs": {"n_results": 1, "search_distance": 0.9},
            }
        )
        response = res["answer"]
        chat_history.append((message, response))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
