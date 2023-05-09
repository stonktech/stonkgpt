from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import SerpAPIWrapper
from langchain.utilities import GoogleSearchAPIWrapper

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from prompts import auto_score_prompt
import re

llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo")

# prompt definition
inference_prompt = PromptTemplate(template=auto_score_prompt['inference_prompt'],
                                  input_variables=["question", "answerStonk", "answerRaw"])


# main logic for history reference agent
def run_scoring(question, answerStonk, answerRaw):
    # apply to the chain
    array1 = [0]
    array2 = [0]
    while len(array1) != 6 or len(array2) != 6:
        llm_chain = LLMChain(prompt=inference_prompt, llm=llm)
        result = llm_chain.predict(question=question,answerStonk = answerStonk,answerRaw= answerRaw)
        print("Result:", result)
        # Define the regular expression pattern
        pattern = r"\[(.*?)\]"

        # Find all matches in the string and extract the arrays
        arrays = re.findall(pattern, result)

        # Convert the arrays to lists of integers
        array1 = [int(x) for x in arrays[0].split(", ")]
        array2 = [int(x) for x in arrays[1].split(", ")]
        print("Trial:", array1, array2)
    print("Final:", array1, array2)
    return [array1, array2]
