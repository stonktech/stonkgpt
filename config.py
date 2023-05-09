
class Config:
    # store all the configuration variables
    
    def __init__(self):
        self.open_ai_key = ""
        self.serper_api_key = ""
        self.google_api_key = ""
        self.google_cse_id = ""
        self.temperature = 0.2
        self.user_agent_header =  {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"}
        self.fast_llm_model = "gpt-3.5-turbo"
        self.use_azure = False
        self.query_num = 3


