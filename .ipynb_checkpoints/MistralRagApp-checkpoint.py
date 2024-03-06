import shutil
import requests
from urllib.parse import urlparse
import sys
import pandas as pd
from typing import Optional, List, Tuple
from langchain_core.language_models import BaseChatModel
import json
import datasets
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFaceHub
import os
import random
import time
from datasets import Dataset, DatasetDict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import config 




pdfPath = config.pdfPath



if pdfPath is None:
    raise ValueError("pdfPath is None. Please set the  pdf path in config.py.")


##Loading PDF
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
model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
d_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map=d_map)
tokenizer = AutoTokenizer.from_pretrained(model_id)


##Creating base Model Chain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task = "text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=100,
    pad_token_id=tokenizer.eos_token_id,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# prompt_template = """
# <|system|>
# Answer the question based on your knowledge. Use the following context to help:

# {context}

# </s>
# <|user|>
# {question}
# </s>
# <|assistant|>

#  """

# prompt_template = """
# ### [INST] 
# Instruction: Answer the question based on your Knowledge. Here is context to help:

# {context}

# ### QUESTION:
# {question} 


# [/INST]
#  """

prompt_template = """
###Instruction: Answer the question based on your Knowledge. Here is context to help:

### Context:
{context}

### Question:
{question} 

### Answer:"""




prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm_chain = LLMChain(llm=llm, prompt=prompt)


##Creating Context Chain
from langchain_core.runnables import RunnablePassthrough

retriever = db.as_retriever()

rag_chain = (
 {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
)


import gradio as gr
pattern = r"[^\w\s,.'\)\"]" 
def predict(type, question):
    if type == "Base":
        ans = llm_chain.invoke({"context":"", "question": question})
    else:
        ans = rag_chain.invoke(question)
       
    ans = ans["text"]
    splits = re.split(pattern, ans)
    ans = splits[0]
    return ans    

pred = gr.Interface(
    fn=predict,
    inputs=[
        gr.Radio(['Base', 'Context'], label="Select One"),
        gr.Textbox(label="Question"),
    ],
    outputs="text",
    title="Retrieval Augumented Generation using Mistral7B"
)

pred.launch(share=True)





























