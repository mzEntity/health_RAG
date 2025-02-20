import os
import utils
import json

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate


class ReformatManager:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
            base_url=os.environ["OPENAI_BASE_URL"],
            temperature=0.6
        )

        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=False
        )
        
        self.set_prompt()


    def set_prompt(self):
        template = """\
Follow the steps to complete the following tasks:
1. The following content is from the user's question input to the medical intelligent question and answer system, please convert these contents into a readable, concise and accurate form. If it already satisfies these properties, there is no need for conversion. You can leave out some irrelevant information, but you must make sure that the rest is true to the original text.
2. If a paragraph consists of many questions, divide them into sub-questions.When splitting subproblems, the context must be completed for each subproblem to ensure that no information is missing when treating each subproblem individually. In particular, nouns such as drugs, symptoms, diseases, etc. must appear in each sub-problem.
3. Each subquestion needs to contain full information about the disease, symptoms, and so on
4. Questions need to be rigorous, and there should be no useless information such as emotional content, patient catchphrases, etc. Follow the "what", "how" and so on templates
5. Answer the succinct questions just generated

Example Process:

User Question: "医生，我最近感觉身体有些不适，尤其是晚上总是难以入睡，而且白天也常常觉得疲惫，这种情况已经持续了好几个星期了，您能告诉我这可能是什么原因吗？还有，我该如何改善这种状况呢？"

Response: 
{
    "pairs": [
    {
        "question": "晚上失眠、白天疲惫的原因是什么？",
        "answer": "可能由压力、焦虑或不规律生活引起。"
    },
    {
        "question": "晚上失眠、白天疲惫应该如何治疗？",
        "answer": "建立规律作息；创造良好睡眠环境；避免刺激性饮料；进行放松活动；如持续，咨询医生。"
    }
    ]
}

Note: Result should be in Chinese.

Current user question: {$user_question$}

The output should be a markup snippet formatted in the following mode, including "```json" at the beginning and "```" at end:
{
     "pairs": [
       {
         "question": "The question after simplification",
         "answer": "Concise and useful answers based on the user's questions."
       },
       // More questions and Answers...
     ]
   }
"""

        self.prompt = template


    def reformat(self, question):
        prompt = self.prompt

        message = prompt.replace("$user_question$", question)
        response = self.conversation.predict(input=message)
        
        # messages = prompt.format_messages(user_question=question)
        # response = self.conversation.predict(input=messages[0].content)
        
        try:
            response = response.strip().lstrip("```json").rstrip("```")
            parsed_data = json.loads(response)
            return parsed_data["pairs"]
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {response}")
            return None
    


class RejectManager:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
            base_url=os.environ["OPENAI_BASE_URL"],
            temperature=0.6
        )

        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=False
        )
        
        self.set_schema()
        self.set_prompt()

    def set_schema(self):
        self.deter_schema = ResponseSchema(name="belong", description="whether the following issues fall within the domain of health care.(True or False)")
    
        self.response_schemas = [self.deter_schema]
        
        self.output_parser = StructuredOutputParser(response_schemas=self.response_schemas)        
        self.format_instructions = self.output_parser.get_format_instructions()


    def set_prompt(self):
        template = """\
Determine whether the following issues fall within the domain of health care.
It doesn't have to be particularly harsh, as long as the problem is in a medical or health context.
Example Process:
User Question: "医生，我最近感觉身体有些不适，尤其是晚上总是难以入睡，而且白天也常常觉得疲惫，这种情况已经持续了好几个星期了，您能告诉我这可能是什么原因吗？还有，我该如何改善这种状况呢？"
Response: 
- "belong": "True"


User Question: "蒸汽机是谁发明的？在什么时间发明的？"
Response: 
- "belong": "False"


User Question: "严重吗？"
Response: 
- "belong": "True"
explanation: This phrase may arise when a doctor tells a patient about a condition and the patient asks about its severity


User Question: "如何治疗？"
Response: 
- "belong": "True"
explanation: This phrase may arise when a patient asks about treatment after a doctor has told him or her about a condition


Current user question: {user_question}

{format_instructions}
"""

        self.prompt = ChatPromptTemplate.from_template(template=template)


    def determine(self, question):
        prompt = self.prompt

        messages = prompt.format_messages(user_question=question, 
                                          format_instructions=self.format_instructions)
        
            
        response = self.conversation.predict(input=messages[0].content)
        output_dict = self.output_parser.parse(response)
        
        return output_dict['belong']

class HistoryManager:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
            base_url=os.environ["OPENAI_BASE_URL"],
            temperature=0.6
        )

        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=False
        )
        
        self.set_schema()
        self.set_prompt()

    def set_schema(self):
        self.quest_schema = ResponseSchema(name="question", description="Full, independent question")
    
        self.response_schemas = [self.quest_schema]
        
        self.output_parser = StructuredOutputParser(response_schemas=self.response_schemas)        
        self.format_instructions = self.output_parser.get_format_instructions()

    def set_prompt(self):
        template = """\
Determine whether the user's question is missing key information such as context. If it is missing, supplement the question into a complete context-appropriate question based on the historical information provided. If the question itself is complete, leave it as is.
This question can be asked on its own, and all required information must be retained. If necessary, extract key medical and health terms from the history and fill in the omitted sections.
The history questions here are listed in chronological order, and the later question should have a higher weight because it is closer to the current question in time.
Complete pronouns such as "this", "that" and completely omitted subject and object information.
Example Process:
User Question: "医生，我最近感觉身体有些不适，尤其是晚上总是难以入睡，而且白天也常常觉得疲惫，这种情况已经持续了好几个星期了，您能告诉我这可能是什么原因吗？还有，我该如何改善这种状况呢？"
history: ""
Response: 
- "question": "医生，我最近感觉身体有些不适，尤其是晚上总是难以入睡，而且白天也常常觉得疲惫，这种情况已经持续了好几个星期了，您能告诉我这可能是什么原因吗？还有，我该如何改善这种状况呢？"
explanation: "The problem already has enough context information, so leave it as is."


User Question: "头痛如何治疗？"
history: "艾滋病如何防治？"
Response: 
- "question": "头痛如何治疗？"
explanation: "Even if its historical information is different from what the current issue is discussing, the issue already has enough context to leave it as it is"


User Question: "如何确诊可能的疾病？"
history: "头疼脑热可能的原因是什么？"
Response: 
- "question": "头疼脑热需要哪些检查来确诊？"
explanation: "How to check for this problem There is no symptom or disease information, so look in the history."


Current user question: {user_question}
history: {history}

{format_instructions}
"""

        self.prompt = ChatPromptTemplate.from_template(template=template)


    def expand(self, question, history):
        prompt = self.prompt
        history_str = ' '.join(history)
        # print(history_str)
        messages = prompt.format_messages(user_question=question, history=history_str,
                                          format_instructions=self.format_instructions)
        
            
        response = self.conversation.predict(input=messages[0].content)
        output_dict = self.output_parser.parse(response)
        
        return output_dict['question']    
    
if __name__ == "__main__":
    utils.setup()
    
    print("正在连接大模型...")
    questionManager = ReformatManager()
    determine = RejectManager()
    history = HistoryManager()
    h = []
    
    while True:
        user_question = input("> ")
        if user_question == "quit":
            break
        
        deter = determine.determine(user_question)
        if deter == "True":
            print(f"属于, {deter}")
            pairs = questionManager.reformat(user_question)
            for pair in pairs:
                print(f"simplified:{pair['question']}\nanswer:{pair['answer']}")
                print(h)
                new_quest = history.expand(pair['question'], h)
                print(new_quest)
                h.append(new_quest)
                
        else:
            print(f"不属于, {deter}")
        
        print()