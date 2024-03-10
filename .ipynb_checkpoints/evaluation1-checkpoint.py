import torch
import shutil
from urllib.parse import urlparse
import sys
import pandas as pd
from langchain_core.language_models import BaseChatModel
import json
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models import ChatHuggingFace
from langchain_core.language_models import BaseChatModel
import os
import csv
from datasets import Dataset, DatasetDict

os.environ['OPENAI_API_KEY'] = ""
os.environ['HUGGINGFACEHUB_API_TOKEN'] = ""


EVALUATION_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback:"""



from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage
evaluation_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a fair evaluator language model."),
        HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT),
    ]
)





# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
eval_chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
evaluator_name = "GPT4"



answer_path = "evalDatasets/stocksQAWithZephr_Finetuned.csv"
df = pd.read_csv(answer_path) 
answers = Dataset.from_pandas(df)
    
i = 1
answersWithEvaluationScores = []
for experiment in answers:
    
    print(f"Evaluation datapoint {i}/{len(answers)} ......................")
    
    i = i + 1
    
    eval_prompt = evaluation_prompt_template.format_messages(
        instruction=experiment["question"],
        response=experiment["llmAnswer"],
        reference_answer=experiment["correctAnswer"],
    )


    eval_result = eval_chat_model.invoke(eval_prompt)
    feedback, score = [item.strip() for item in eval_result.content.split("[RESULT]")]

    
    answersWithEvaluationScores.append(
            {
                "question": experiment["question"],
                "llmScore": score,
            }
        )
    df = pd.DataFrame.from_dict(answersWithEvaluationScores)
    df.to_csv("Scores/stocksQAWithZephr_FinetunedScores.csv", index=False)