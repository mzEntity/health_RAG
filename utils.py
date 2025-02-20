import os
from dotenv import load_dotenv, find_dotenv
import openai

def setup():
    _ = load_dotenv(find_dotenv(), override=True)
    # avoid api_connection_error when using llama_index
    # stackoverflow: 76452544
    openai.api_key = os.environ['OPENAI_API_KEY']
    openai.base_url = os.environ['OPENAI_BASE_URL']
    
class Logger:
    def __init__(self):
        pass
    
    @staticmethod
    def log(filename, message):
        try:
            with open(filename, 'a', encoding='utf-8') as file:
                file.write(message + '\n')
        except Exception as e:
            print(f"Error logging message: {e}")

if __name__ == "__main__":
    Logger.log('logfile.txt', '这是一个日志消息')
    