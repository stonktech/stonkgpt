from tools.magic_ai import call_ai_function
from config import Config
from prompts import Ray_res_JSON_SCHEMA, Ray_Dalio_GPT, future_prediction_prompt
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from tools.llm_utils import create_chat_completion
from tools.spinner import Spinner
import json
import datetime


cfg = Config()

# define variables
messages = []
thoughts = ""
event_analysis = {}
# inferece_prompt = PromptTemplate(template=history_reference_prompt['inference_prompt'], input_variables=["question", "event_description"])


def run_history_inference(historical_events: list, factor_inferences: list) -> dict:
    # Try to fix the JSON using GPT:
    function_string = "def fix_json(historical_events: list, factor_inferences:list) -> dict:"
    args = [f"'''{historical_events}'''", f"'''{factor_inferences}'''"]
    description_string = (
        "this function contains a highly intelligent AI chatbot called econGPT, it can analyze different factors in historical event and output a detailed and informative description of the event.\n"
        "two list will be the input of this function, one will be the historical events, the other will be the factor inferences.\n"
        " the output return will be a python dictionary or json, the key will be historical event, the value is econGPT's summary of the event with respect to all the factors listed in the factor inference list, the summary should contains the following information: \n" 
        "1. how did the each of the relevant factors affect the event, give VERY detailed information\n"
        "2. what was the later consequence of event if econGPT think necessary\n"
        "3. the description of event should contain qualitivie and quantitive detailed information\n\n"
        " The return is always in json format, and the value is always a string."
    )

    result_string = call_ai_function(
        function_string, args, description_string, model=cfg.fast_llm_model
    )
    return result_string

def fix_json(json_str: str, schema: str) -> str:
    """Fix the given JSON string to make it parseable and fully compliant with the provided schema."""
    # Try to fix the JSON using GPT:
    function_string = "def fix_json(json_str: str, schema:str=None) -> str:"
    args = [f"'''{json_str}'''", f"'''{schema}'''"]
    description_string = (
        "Fixes the provided JSON string to make it parseable"
        " and fully compliant with the provided schema.\n If an object or"
        " field specified in the schema isn't contained within the correct"
        " JSON, it is omitted.\n This function is brilliant at guessing"
        " when the format is incorrect."
    )

    # If it doesn't already start with a "`", add one:
    if not json_str.startswith("`"):
        json_str = "```json\n" + json_str + "\n```"
    result_string = call_ai_function(
        function_string, args, description_string, model=cfg.fast_llm_model
    )
    return result_string

def init_message() -> str:
    """Initialize the message for the history reference prompt."""
    initial_message = {
        "role": "system",
        "content": Ray_Dalio_GPT['starter_prompt'],
    }
    # convert to string
    current_date_time = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    date_time_message = {
        "role": "system",
        "content": f"The current time and date is {current_date_time}",
    }
    messages.append(initial_message)
    messages.append(date_time_message)

def add_user_message(question: str) -> str:
    """Add a user message to the history reference prompt."""
    message = "Given above response format and thought process, answer question belwo: \nQuestion: " + question 

    user_message = { "role": "user", "content": message }
    messages.append(user_message)

def relevant_event_extractor(event_str: str) -> str:
    # given " - The bankruptcy of Lehman Brothers in 2008\n- The bankruptcy of Washington Mutual in 2008\n- The bankruptcy of IndyMac Bank in 2008"
    # return ["The bankruptcy of Lehman Brothers in 2008", "The bankruptcy of Washington Mutual in 2008", "The bankruptcy of IndyMac Bank in 2008"]
    event_str = event_str.replace("- ", "")
    return event_str.split("\n")

def get_event_analysis(event_list, references):

    base_system_prompt = {
        "role": "system",
        "content": Ray_Dalio_GPT['event_analysis_prompt'],
    }
    for event in event_list:

        print("extracting event:", event)
        messages = []
        messages.append(base_system_prompt)
        event_description = {
            "role": "user",
            "content": str({"event": event, "reference_list": references}),
        }
        messages.append(event_description)
        result = create_chat_completion(model=cfg.fast_llm_model, messages=messages, temperature=0.3)
        event_analysis[event] = result

    return event_analysis


def final_conclusion_generation(reference, event_description, question):
    # generate the final conclusion

    # final_message = []
    final_conclusion = {
        "role": "system",
        "content": Ray_Dalio_GPT['final_response_prompt'],
    }
    custom_user_message = {
        "role": "user",
        "content": str({"reference": reference, "event_description": event_description, "question": question}),
    }

    messages.append(final_conclusion)
    messages.append(custom_user_message)
    result = create_chat_completion(model=cfg.fast_llm_model, messages=messages, temperature=0.3)
    return result

def construct_reference(event_result):
    # construct the reference
    reference_str = "Reference: \n"
    for event, res in event_result.items():
        reference_str += f"{event}: {res}\n"

    return reference_str

def run_ray_dalio(question: str, user_input = True) -> dict:
    """Run the Ray Dalio prompter."""

    # initialize the message
    init_message()
    print("initilizing process...")
    # add user message
    add_user_message(question)

    with Spinner("Thinking... "):
        result = create_chat_completion(model=cfg.fast_llm_model, messages=messages, temperature=0.3)

        # Fix the JSON:
        intermediate_result = fix_json(result, Ray_res_JSON_SCHEMA)
        intermediate_result = json.loads(intermediate_result)['thoughts']

    print("intermediate result: ", intermediate_result)

    # relevant event extraction
    relevant_events = relevant_event_extractor(intermediate_result['related_event'])
    reference_factor = relevant_event_extractor(intermediate_result['reference_factor'])
    print("relevant events: ", relevant_events)
    print("reference factor: ", reference_factor)

    # relevant_events =  ['The bankruptcy of Lehman Brothers in 2008', 'The bankruptcy of Washington Mutual in 2008', 'The bankruptcy of IndyMac Bank in 2008']
    # reference_factor = ['- Availability of credit', '- Financial stability of the tech sector', '- Government policies and response']
    result = get_event_analysis(relevant_events, intermediate_result['reference_factor'])

    reference_prompt = construct_reference(result)

    event_description = intermediate_result['event_of_interest'] + "\n\n"
    # collect data
    if user_input:
        for factor in reference_factor:
            factor_data = input("Please provide a more detail of " + factor + "in "  + intermediate_result['event_of_interest'] + ": \n")
            event_description += factor + ": " + factor_data + "\n"

    with Spinner("Thinking... "):
        # final_conclusion_generation
        final_conclusion = final_conclusion_generation(reference_prompt, event_description, question)
        final_conclusion = fix_json(final_conclusion, Ray_res_JSON_SCHEMA)
        final_conclusion = json.loads(final_conclusion)['thoughts']

    
    # constructing the final result
    final_result = "Question: " + question + "\n\n"
    final_result += "My initial thought: " + intermediate_result['text'] + "\n\n"
    final_result += "I will analyze " + intermediate_result['event_of_interest'] + " in the context of \n" + intermediate_result['reference_factor'] + "\n\n"
    final_result += "I found those similar event in history: \n\n"
    for event, res in result.items():
        final_result += f"{event}:\n {res}\n\n"

    final_result += "\n\n"
    final_result += "My final conclusion: " + final_conclusion['text']
    final_result += "\n\nHowever, we still need to watch out that " + final_conclusion['criticism']


    return final_result

# How would bankruptcy of silicon valley bank affect our economic and what action the federal government will take? 

# the credit is severely tighten up due to federal reserve increased interest rate, there are limited credit for startup and technology sector

# the overall financial system is not as stable as before because the inflation is so high, the federal reserve has to increase interest rate to fight the inflation, this cause the small bank at risk, and silicon valley bank has large asset in bond, which decrease the value.

# we don't know yet, I want you to tell me what action would federal government do in response to the bankruptcyin Bankruptcy, would they bail the bank out

