import shutil
import requests
import sys
from typing import Optional, List, Tuple
from langchain_core.language_models import BaseChatModel
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFaceHub
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import config 


##Loading the file path
pdfPath = config.pdfPath


if pdfPath is None:
    raise ValueError("pdfPath is None. Please set the  pdf path in config.py.")

##Loading the text file
loader = PyPDFLoader(pdfPath)



##splitting the text file
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,  
        chunk_overlap=60,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )
try:
    langchain_docs = loader.load_and_split(text_splitter=text_splitter) #loads and slits
    #docs = loader.load()
    #langchain_docs = text_splitter.split_documents(docs)
except Exception as e:
    raise ValueError("An error occurred:", e)



##Loading the embedding Model
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embeddingModelName = "BAAI/bge-base-en-v1.5"

embeddingModel = HuggingFaceEmbeddings(model_name=embeddingModelName)

try:
    db = FAISS.from_documents(langchain_docs, embeddingModel)
except Exception as e:
    raise ValueError("An error occurred:", e)




##Loading the Model to answer questions
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = 'HuggingFaceH4/zephyr-7b-beta'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

try:
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except Exception as e:
    raise ValueError("An error occurred:", e)



##Creating base Model Chain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=200,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

prompt_template = """
<|system|>
Answer the question based on your knowledge. Use the following context to help:

{context}

</s>
<|user|>
{question}
</s>
<|assistant|>

 """

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm_chain = prompt | llm | StrOutputParser()




##Creating Context Chain
from langchain_core.runnables import RunnablePassthrough

retriever = db.as_retriever()

rag_chain = (
 {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
)



import gradio as gr

def predict(type, question):
    if type == "Base":
        ans = llm_chain.invoke({"context":"", "question": question})
        return ans
    else:
        ans = rag_chain.invoke(question)
        return ans    

pred = gr.Interface(
    fn=predict,
    inputs=[
        gr.Radio(['Base', 'Context'], label="Select One"),
        gr.Textbox(label="Question"),
    ],
    outputs="text",
    title="Retrieval Augumented Generation using zephyr-7b-beta"
)

pred.launch(share=True)











