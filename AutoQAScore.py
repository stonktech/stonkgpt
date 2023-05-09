import numpy as np
import pandas as pd
from LangChain.classification_prompter import get_question_type
from LangChain.history_reference_agent import run_history_inference
from LangChain.stock_specific_prompter import run_stock_specific
from LangChain.prediction_focused_prompter import run_future_prediction
from LangChain.ray_dalio_prompter import run_ray_dalio
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from LangChain.auto_grade_prompter import run_scoring

if __name__ == "__main__":

    '''
        history_reference: questions that ask for a comparison between two or more events in history
        stock_specific: questions that ask for a prediction about a specific stock
        prediction_focused: questions that ask for a prediction about the market as a whole
        other: questions that do not fit into any of the above categories
    '''

    categorization = False
    questions = [
# "How would the bankruptcy of Silicon Valley Bank affect our economy and what action will the federal government take?",
# "One of the biggest crypto exchanges, FTX, has collapsed. How will that affect the crypto market and is it a good chance for me to invest in bitcoin?",

## "Elon Musk has bought twitter and laid-off 70% of Twitter’s employees, how will that influence twitter?",
## "If the FED decides not to increase the interest rate this May, how will that influence the inflation and the market?",

#"The gold price reached a historical high, can you give me the prediction of the market move and provide me with some suggestions?",
#"With the recent merger between two major financial institutions, what can we expect in terms of market share, competition, and possible regulatory actions?",

## "The European Central Bank has announced new quantitative easing measures. How will these measures impact the European financial markets, and what could be the potential spillover effects on the global economy?",

#"The U.S. government has imposed new trade tariffs on several countries, which could lead to a trade war. What sectors might be most impacted, and how might this affect the overall stock market performance?",
#"A major airline has filed for bankruptcy protection. What are the potential consequences for the travel industry and the stock prices of other airlines? Are there any investment opportunities in this situation?",

##"The housing market is showing signs of a potential bubble. What are the key indicators, and how could a possible burst affect the broader economy and the stock market?",

#"There have been growing concerns about the high level of corporate debt. How could a potential wave of defaults impact the financial markets, and which sectors might be most vulnerable?",
#"Oil prices have experienced significant volatility in recent months. What factors are driving these price fluctuations, and how could they influence the performance of energy stocks and the broader market?",

#"The U.S. dollar has been weakening against other major currencies. How could this trend affect U.S. exports, domestic inflation, and the performance of multinational corporations?",
##"A new technology has been introduced in the financial sector, with the potential to disrupt traditional banking services. How might this innovation impact the industry, and what investment opportunities could arise from this development?",

"Recent changes in tax policy have led to speculation about the potential impact on the stock market. How might these changes affect corporate earnings and investment strategies in the short and long term?"
]
    #question = input("Please enter your question: ")
    cols = ["Question", "Answer", "Similarity detection", "Causal reasoning", "Future scenario generation", "Clarity and coherence", "Novelty and creativity", "Adaptability and responsiveness"]
    scores_Stonk = pd.DataFrame(columns=cols)
    scores_Raw = pd.DataFrame(columns=cols)
    for question in questions:
        print("Question:", question)
        flag = True
        ii = 0
        while flag:
            print("Trial:", ii)
            try:
                llm = ChatOpenAI(temperature=0.8, model_name="gpt-3.5-turbo")
                template = """{question}"""
                prompt = PromptTemplate(template=template, input_variables=["question"])

                llm_chain = LLMChain(prompt = prompt, llm=llm)
                rawGPT = llm_chain.predict(question=question)
                print("RAWGPT", rawGPT)

                stonkGPT = run_ray_dalio(question, False)
                print("StonkGPT", stonkGPT)
                flag = False
            except Exception as e:
                print("Error", e)
                flag = True
            ii += 1
        scores = run_scoring(question, stonkGPT, rawGPT)
        print("Scores:", scores)
        res = [question, rawGPT]
        for i in range(6):
            res.append(scores[0][i])
        new_row = pd.DataFrame([res], columns=cols)
        scores_Stonk = pd.concat([scores_Stonk, new_row], ignore_index=True)

        res = [question, stonkGPT]
        for i in range(6):
            res.append(scores[1][i])
        new_row = pd.DataFrame([res], columns=cols)
        scores_Raw = pd.concat([scores_Raw, new_row], ignore_index=True)
    scores_Stonk.to_csv("scores_Stonk.csv", index=False)
    scores_Raw.to_csv("scores_Raw.csv", index=False)
            #break
# split trigger into categories