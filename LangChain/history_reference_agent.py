from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import SerpAPIWrapper
from langchain.utilities import GoogleSearchAPIWrapper

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from prompts import history_reference_prompt

import os
from config import Config

cfg = Config()
# define environment variables
os.environ["GOOGLE_CSE_ID"] = cfg.google_cse_id
os.environ["GOOGLE_API_KEY"] = cfg.google_api_key
os.environ["SERPAPI_API_KEY"] = cfg.serper_api_key


wikipedia = WikipediaAPIWrapper()
serp_search = SerpAPIWrapper()
google_search = GoogleSearchAPIWrapper()
llm = ChatOpenAI(temperature=0.8, model_name="gpt-3.5-turbo")

# prompt definition
extraction_prompt = PromptTemplate(template=history_reference_prompt['history_event_extraction_prompt'], input_variables=["question"])
inferece_prompt = PromptTemplate(template=history_reference_prompt['inference_prompt'], input_variables=["question", "event_description"])

# this function will take in a history reference question and return the relevant all relevant history events in an array
def get_history_events(question):
    
    llm_chain = LLMChain(prompt=extraction_prompt, llm=llm, verbose=True)
    result = llm_chain.run(question)
    event_list = result.split(",")
    return event_list

# given a list event, search them up in google one by one
def search_event(event_list):
    event_dict = {}
    for event in event_list:
        event_dict[event] = google_search.run(event)

    return event_dict

def search_event_prompt_format(event_dict):

    prompt = "Here is detailed information about the events mentioned: \n"

    for event, result in event_dict.items():
        prompt += "Event: " + event + "\n"
        prompt += "Event Description: " + result + "\n\n"

    return prompt
    

# main logic for history reference agent
def run_history_inference(question):

    # get the history events
    event_list = get_history_events(question)
    print("extracted events: ", event_list)

    # search up the events
    event_dict = search_event(event_list)
    print("searched events information...")

    # format the prompt
    event_prompt = search_event_prompt_format(event_dict)
    print("formatted prompt...")

    # apply to the chain
    llm_chain = LLMChain(prompt=inferece_prompt, llm=llm)
    print("finalized response...")
    result = llm_chain.predict(question=question, event_description=event_prompt)

    return result
