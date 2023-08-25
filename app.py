import chainlit as cl
from langchain.llms import CTransformers
from langchain.chains import LLMChain, RetrievalQA
from langchain import PromptTemplate

prompt_template = """You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only 
respond once as 'Assistant' 
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=['question'])
    return prompt



def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q4_1.bin",
        model_type="llama",
        max_new_tokens=800,
        temperature=0.7
    )

    return llm
@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=load_llm(), verbose=True)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Do any post processing here

    # "res" is a Dict. For this chain, we get the response by reading the "text" key.
    # This varies from chain to chain, you should check which key to read.
    await cl.Message(content=res["text"]).send()
