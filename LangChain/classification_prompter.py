from langchain import PromptTemplate, FewShotPromptTemplate
from prompts import classification_prompt

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain

from config import Config
import os

cfg = Config()
# define environment variables
os.environ['OPENAI_API_KEY'] = cfg.open_ai_key


# specify the template to format the examples we have provided.
# use the `PromptTemplate` class for this.
example_formatter_template = """
Question: {question}
Question Type: {question_type}\n
"""

example_prompt = PromptTemplate(
    input_variables=["question", "question_type"],
    template=example_formatter_template,
)

# create the `FewShotPromptTemplate` object.
few_shot_prompt = FewShotPromptTemplate(
    # These are the examples we want to insert into the prompt.
    examples=classification_prompt['examples'],

    # This is how we want to format the examples when we insert them into the prompt.
    example_prompt=example_prompt,

    # The prefix is some text that goes before the examples in the prompt.
    # this consists of intructions.
    prefix=classification_prompt['general_prompt'],

    # The suffix is some text that goes after the examples in the prompt.
    # this is where the user input will go
    suffix="Question: {input}\nQuestion Type:",

    # The input variables are the variables that the overall prompt expects.
    input_variables=["input"],

    # The example_separator is the string we will use to join the prefix, examples, and suffix together with.
    example_separator="\n",
)

chat = ChatOpenAI(temperature=0.5)
chain = LLMChain(llm=chat, prompt=few_shot_prompt, verbose=True)

# generate a prompt using the `format` method.
def get_prompt(question):
    return few_shot_prompt.format(input=question)

def get_question_type(question):
    return chain.predict(input=question)

