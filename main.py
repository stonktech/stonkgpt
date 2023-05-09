from LangChain.classification_prompter import get_question_type
from LangChain.history_reference_agent import run_history_inference
from LangChain.stock_specific_prompter import run_stock_specific
from LangChain.prediction_focused_prompter import run_future_prediction
from LangChain.ray_dalio_prompter import run_ray_dalio




if __name__ == "__main__":

    '''
        history_reference: questions that ask for a comparison between two or more events in history
        stock_specific: questions that ask for a prediction about a specific stock
        prediction_focused: questions that ask for a prediction about the market as a whole
        other: questions that do not fit into any of the above categories
    '''

    categorization = False
    question = input("Please enter your question: ")

    if categorization:
        question_type = get_question_type(question)

        print("Question type: ", question_type)

        if question_type == "history_reference":
            print(run_history_inference(question))

        elif question_type == "stock-specific":
            print(run_stock_specific(question))

        elif question_type == "prediction-focused":
            print(run_future_prediction(question))

        else:
            print("Sorry, I'm afraid I don't know the answer to that question. Please try asking another question.")

    else:
        print(run_ray_dalio(question))

# split trigger into categories 