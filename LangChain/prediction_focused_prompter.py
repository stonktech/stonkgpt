from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import SerpAPIWrapper
from langchain.utilities import GoogleSearchAPIWrapper

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from prompts import future_prediction_prompt

import os
from config import Config

cfg = Config()
# define environment variables

# define language model
llm = ChatOpenAI(temperature=0.8, model_name="gpt-3.5-turbo")

# define prompt
inferece_prompt = PromptTemplate(template=future_prediction_prompt['inference_prompt'], input_variables=["question"])

def run_future_prediction(question):

    llm_chain = LLMChain(prompt=inferece_prompt, llm=llm)
    print("finalized response...")
    result = llm_chain.predict(question=question)

    return result
