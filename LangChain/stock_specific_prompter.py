from langchain.utilities import SerpAPIWrapper
from langchain.utilities import GoogleSearchAPIWrapper

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from prompts import Stock_specific_prompt
from tools.web_search import google_official_search, google_search, browse_website

import os
from config import Config

cfg = Config()
# define environment variables
os.environ["GOOGLE_CSE_ID"] = cfg.google_cse_id
os.environ["GOOGLE_API_KEY"] = cfg.google_api_key
os.environ["SERPAPI_API_KEY"] = cfg.serper_api_key


serp_searcher = SerpAPIWrapper()
google_searcher = GoogleSearchAPIWrapper()
llm = ChatOpenAI(temperature=cfg.temperature, model_name="gpt-3.5-turbo")

# prompt definition
question_query_prompt = PromptTemplate(template=Stock_specific_prompt['question_query_prompt'], input_variables=["query_number", "stock", "question"])
stock_extraction_prompt = PromptTemplate(template=Stock_specific_prompt['stock_extraction_prompt'], input_variables=["question"])
inference_prompt = PromptTemplate(template=Stock_specific_prompt['inference_prompt'], input_variables=["question", "reference"])
# this function will take in a history reference question and return the relevant all relevant history events in an array
def stock_extraction(question):
    
    llm_chain = LLMChain(prompt=stock_extraction_prompt, llm=llm, verbose=True)
    stock_result = llm_chain.run(question)
    return stock_result


def question_query_prompt_wrapper(query_number, question, stock):
    
    llm_chain = LLMChain(prompt=question_query_prompt, llm=llm, verbose=True)
    result = llm_chain.predict(question=question, query_number=query_number, stock=stock)
    
    question_list = result.split("\n")
    return [question[2:] for question in question_list]
        
def search_question_summarized_query(question_list):
    question_dict = {}
    for question in question_list:
        link_list = google_official_search(question)

        answer = ""

        for link in link_list[:1]:
            answer += browse_website(link, question) + "\n"

        question_dict[question] = answer

    return question_dict

def search_question_query(question_list):
    question_dict = {}
    for question in question_list:
        question_dict[question] = google_search(question)

    return question_dict

def search_question_query_format(question_dict):
    prompt = "Reference from google search (just use this information if you found necessary, not required) \n"

    for question, result in question_dict.items():
        prompt += "question: " + question + "\n"
        prompt += "result: " + result + "\n\n"

    return prompt


# main logic for history reference agent
def run_stock_specific(question):

    # get the history events
    stock_result = stock_extraction(question)
    print("extracted events: ", stock_result)

    query_num = cfg.query_num
    # search up the events
    question_query = question_query_prompt_wrapper(query_num, question, stock_result)
    
    print("searched events information...")
    
    #search_result = search_question_summarized_query(question_query)
    search_result = search_question_query(question_query)

    # format the search result
    search_result_prompt = search_question_query_format(search_result)

    # apply to the chain
    llm_chain = LLMChain(prompt=inference_prompt, llm=llm)
    print("finalized response...")
    result = llm_chain.predict(question=question, reference=search_result_prompt)

    return result
