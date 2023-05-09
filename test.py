from LangChain.classification_prompter import get_question_type
from LangChain.history_reference_agent import run_history_inference
from LangChain.stock_specific_prompter import stock_extraction, run_stock_specific
from tools.web_search import google_search, google_official_search, browse_website
from LangChain.prediction_focused_prompter import run_future_prediction



question = "how would apple stock perform if they released a new iphone in 2024?"
question2 = "Considering the shift towards renewable energy and electric vehicles, how do you predict the stock performance of traditional oil and gas companies in the next decade compared to renewable energy companies"

question3 = "given current market situation, will inflation be a problem in the next 5 years?"
question4 = "What were the main reasons behind the Long-Term Capital Management (LTCM) hedge fund collapse in 1998, and can you identify any other hedge fund failures that share similarities in terms of causes and consequences?"
# answer = get_answer(question3)
question5 = "could you find me some relevant history events that are related to the 2008 financial crisis and also 2000 internet bubble?"
# print("result: ", answer)
# test()
# print(run_history_inference(question5))
q1 = "How have changes in the pharmaceutical industry, such as the development of COVID-19 vaccines by Pfizer and Moderna, affected the stock prices of these companies and their competitors?"


if __name__ == "__main__":

    # tes
    # result = google_official_search("what is current apple stock price?")
    # print(result)
    # result = ['https://investor.apple.com/stock-price/default.aspx', 'https://www.marketwatch.com/investing/stock/aapl', 'https://finance.yahoo.com/quote/AAPL/', 'https://markets.businessinsider.com/stocks/aapl-stock', 'https://investor.apple.com/faq/default.aspx', 'https://support.apple.com/guide/iphone/check-stocks-iph1ac0b1bc/ios', 'https://apps.apple.com/us/app/stocks/id1069512882', 'https://www.apple.com/newsroom/2022/04/apple-reports-second-quarter-results/']
    # # question = input("Hi, I am Stonk GPT, I can answer any questions related to market prediction, history inference, and stock specific questions. What would you like to ask me?")
    # content = browse_website(result[5], "what is current apple stock price?")
    # print(content)
    
    print(run_future_prediction("Based on historical trends and current market conditions, which emerging technologies or industries are likely to experience significant growth in the next 5 to 10 years?"))
