
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain, LLMChain

from langchain.llms import LlamaCpp, HuggingFaceTextGenInference
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from deploy.runpod import deploy_llm

def deploy_model():
    endpoint = deploy_llm()
    print(f'Use this endpoint: {endpoint}')

def get_rag_chain(inference_server_url):
    llm = HuggingFaceTextGenInference(
        inference_server_url=inference_server_url,
        stop_sequences=['User:'],
        max_new_tokens=500,
        top_k=10,
        top_p=0.95,
        typical_p=0.95,
        temperature=0.1,
        repetition_penalty=1.03
    )

    prompt_template = PromptTemplate(
        input_variables=['summaries', 'question'],
        template='''{summaries}
User: {question}
Assistant:'''
    )
    loader = TextLoader("data.txt")
    index = VectorstoreIndexCreator().from_loaders([loader])
    retriever = index.vectorstore.as_retriever(
        search_kwargs={'k': 1}
    )

    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        chain_type_kwargs={
            'prompt': prompt_template
        }
    )


def rag(inference_url, question):
    chain = get_rag_chain(inference_url)
    response = chain(question)
    print(response['answer'].strip())
