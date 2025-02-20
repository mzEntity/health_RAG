import os
import utils

from langchain_openai import ChatOpenAI

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate


class UserProfileManager:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
            base_url=os.environ["OPENAI_BASE_URL"],
            temperature=0.6
        )

        self.set_schema()
        self.set_prompt()
        

    def set_schema(self):
        self.gender_schema = ResponseSchema(name="gender", description="user's gender. 男性 or 女性. 未知 if unknown.")
        self.age_schema = ResponseSchema(name="age", description="Age range of users, divided into young(年轻, 18-), middle-aged(中年, 19-60) and elderly(老年, 61+). 未知 if unknown.")
        self.anamnesis_schema = ResponseSchema(name="anamnesis", description="A record of a user's previous health conditions, illnesses, surgeries, and treatments. 未知 if unknown.")
        
        self.response_schemas = [self.gender_schema, self.age_schema, self.anamnesis_schema]
        
        self.output_parser = StructuredOutputParser(response_schemas=self.response_schemas)        
        self.format_instructions = self.output_parser.get_format_instructions()

    def set_prompt(self):
        template = """\
For the following text in Chinese, extract the following information in Chinese: 

1. gender: user's gender. male or female.
2. age: Age range of users, divided into young(年轻, 18-), middle-aged(中年, 19-60) and elderly(老年, 61+).
3. anamnesis: A record of a user's previous health conditions, illnesses, surgeries, and treatments.

All result should be in Chinese.
Example Process:

User Question: "我今年21岁，是男性，2018年被诊断高血压，服用洛卡特普控制血压。"

Response:

  gender: "男性"
  age: "中年"
  anamnesis: "2018年被诊断为高血压，正在服用洛卡特普控制血压"

Text: {text}

{format_instructions}
"""
        self.prompt = ChatPromptTemplate.from_template(template=template)


    def extract(self, text):
        messages = self.prompt.format_messages(text=text, format_instructions=self.format_instructions)
            
        response = self.llm(messages)
        
        output_dict = self.output_parser.parse(response.content)
        return output_dict
    
    
if __name__ == "__main__":
    utils.setup()
    
    print("正在连接大模型...")
    userProfileManager = UserProfileManager()
    
    text = "我叫小李，今年28岁，性别男。我来自北京。我在青少年时期曾经有过哮喘的经历，但随着年龄的增长，症状已经大大减轻，现在几乎没有发作过。"

    resp = userProfileManager.extract(text)
    print(f"text: {text}\nextract:{resp}")
    print()