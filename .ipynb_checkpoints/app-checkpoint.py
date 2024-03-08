##Importing Dependencies.
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
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import config 





##Loading Pdf and Precessing it
pdfPath = config.pdfPath
if pdfPath is None:
    raise ValueError("pdfPath is None. Please set the  pdf path in config.py.")
loader = PyPDFLoader(pdfPath)
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )
try:
    langchain_docs = loader.load_and_split(text_splitter=text_splitter) #loads and slits
    #docs = loader.load()
    #langchain_docs = text_splitter.split_documents(docs)
except Exception as e:
    raise ValueError("An error occurred:", e)

##creating Vector DB
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embeddingModelName = "BAAI/bge-base-en-v1.5"

embeddingModel = HuggingFaceEmbeddings(model_name=embeddingModelName)

db = FAISS.from_documents(langchain_docs, embeddingModel)



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

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)









##Creating base Model Chain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=200,
    pad_token_id=tokenizer.eos_token_id,
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


llm_chain = LLMChain(llm=llm, prompt=prompt)

##Creating Context Chain
from langchain_core.runnables import RunnablePassthrough






##Launching Gradio
import gradio as gr

def predict(type, limit, question):
    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": limit})
    rag_chain = ({"context": retriever, "question": RunnablePassthrough()}| llm_chain)
    if type == "Context":
        ragAnswer = rag_chain.invoke(question)
        context = ragAnswer["context"]
        ans = "Context loaded from most to least in similarity search:"
        i = 1
        for c in context:
            content = c.page_content.replace('\n', ' ')
            ans += "\n\n" + f"context {i}:" + "\n\n" + content
            i += 1
        return ans
        
    if type == "Base":
        ans = llm_chain.invoke({"context":"", "question": question})
        return ans
    else:
        res = rag_chain.invoke(question)
        context = res["context"]
        if len(context) == 0:
            ans = "Please ask questions related to the documents....."
        else:
            ans = res["text"]
        return ans 
           

pred = gr.Interface(
    fn=predict,
    inputs=[
        gr.Radio(['Context', 'BaseModel','RAG'], value = "Context", label="Select Search Type"),
        gr.Slider(0.1, 1, value=0.5, label="Degree of Similarity"),
        gr.Textbox(label="Question"),
    ],
    outputs="text",
    title="Retrieval Augumented Generation using zephyr-7b-beta"
)

pred.launch(share=True)
















