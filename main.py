import utils
from query import QueryManager
from chat import ChatManager
from user import UserProfileManager
from advance import ReformatManager, RejectManager, HistoryManager
from knowledgeGraph.chatbot_graph import ChatBotGraph

from pretty import colored_text
import warnings



if __name__ == "__main__":
    USE_ADVANCED = True
    warnings.filterwarnings('ignore')
    utils.setup()
    
    print("正在加载知识图谱...")
    graphChat = ChatBotGraph()
    
    print("正在加载数据库...")
    queryManager = QueryManager()
    
    print("建立用户数据中...")
    userProfileManager = UserProfileManager()
    
    print("正在连接大模型...")
    
    if USE_ADVANCED:
        rejectManager = RejectManager()
        reformatManager = ReformatManager()
        historyManager = HistoryManager()
        
    chatManager = ChatManager()
    
    # 我叫小李，今年28岁，性别男。我在青少年时期曾经有过哮喘的经历，但随着年龄的增长，症状已经大大减轻，现在几乎没有发作过。除了这个，我的健康状况良好。
    text = input(colored_text("请输入您的性别、年龄和既往病史：", font_color="green"))
    profile = userProfileManager.extract(text)
    print(colored_text(f"个人信息总结：性别{profile['gender']}, {profile['age']}, {profile['anamnesis']}", font_color="cyan"))
    
    utils.Logger.log("./log/chat.log", f"个人信息：{profile}")
    profile["interest"] = ""
    profile["style"] = ""
    
    chatManager.set_userProfile(profile)
    # 医生啊，我最近总是感觉这里不舒服，那里也不对劲，头疼脑热的，吃了药也没见好，你说我这是怎么回事呢？是不是得了什么严重的病啊？需要做哪些检查才能确定下来呢？真的很担心。
    # 医生，我前些天被狗咬了，虽然伤口不大，但我还是很担心会感染狂犬病。我已经打了第一针狂犬病疫苗，但听说要打好多针，这个过程是怎样的？还有啊，打完疫苗后需要注意些什么呢？会不会有什么副作用？我真的有点害怕。
    while True:
        question = input(colored_text("用户：", font_color="green"))
        if question == "quit":
            break
        utils.Logger.log("./log/chat.log", question)
        
        if USE_ADVANCED:
            deter = rejectManager.determine(question)
            utils.Logger.log("./log/chat.log", f"是否属于健康医疗领域：{deter}")
            if deter == "False":
                print(colored_text("本系统只支持回复和健康医疗相关的问题，或者请更改或具体化您提的问题。\n", font_color="red"))
                continue
            expand_question = historyManager.expand(question, chatManager.history)
            utils.Logger.log("./log/chat.log", f"问句改写：{expand_question}")
            
            pairs = reformatManager.reformat(expand_question)
            
            utils.Logger.log("./log/chat.log", f"问句拆分为{len(pairs)}个问题。")
            print(colored_text("模型思考中...\n", font_color="green"))
            for idx, pair in enumerate(pairs):
                simplified_question, brief_ans = pair["question"], pair["answer"]
                utils.Logger.log("./log/chat.log", f"问句{idx+1}：{simplified_question}\n回答{idx+1}: {brief_ans}")
                question = simplified_question
                query = f"{simplified_question} {brief_ans}"
                graph_ret = graphChat.chat_main(simplified_question)
                utils.Logger.log("./log/chat.log", f"知识图谱结果：{graph_ret}")
                reference = queryManager.query_one(query)
                utils.Logger.log("./log/chat.log", f"向量知识库结果：{reference}")
                
                if graph_ret is not None:
                    reference = reference + f"\nknowledgeGraph: {graph_ret}"
                
                extract, response = chatManager.chat_one(question, reference)
                utils.Logger.log("./log/chat.log", f"大模型结果：{response}\n参考知识：{extract}")
                print(colored_text(question, font_color="yellow"))
                print(colored_text(f"大模型：{response}", font_color="white"))
                if extract == "":
                    extract = "无"
                print(colored_text(f"参考知识：\n{extract}\n", font_color="cyan"))
        else:
            query = question
            graph_ret = graphChat.chat_main(question)
            utils.Logger.log("./log/chat.log", f"知识图谱查询结果：{graph_ret}")
            reference = queryManager.query_one(query)
            utils.Logger.log("./log/chat.log", f"向量知识库查询结果：{reference}")
            
            if graph_ret is not None:
                reference = reference + f"\nknowledgeGraph: {graph_ret}"
            
            extract, response = chatManager.chat_one(question, reference)
            utils.Logger.log("./log/chat.log", f"大模型返回结果：{response}\nfrom:{extract}")
            print(colored_text(f"大模型：{response}", font_color="white"))
            if extract == "":
                extract = "无"
            print(colored_text(f"参考知识：\n{extract}", font_color="cyan"))
        
        
        # openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 4096 tokens. However, your messages resulted in 4198 tokens. Please reduce the length of the messages. (request id: 20241021195751114527649ZBCGOtI3) (request id: 2024102119575186572977M6XWC5cc)", 'type': 'invalid_request_error', 'param': 'messages', 'code': 'context_length_exceeded'}}