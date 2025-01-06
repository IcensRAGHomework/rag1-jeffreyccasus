import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

from langchain_core.messages import HumanMessage, SystemMessage

def generate_hw01(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    ai_message = SystemMessage(
            content=[
                {"type": "text", "text": "你目前扮演 台灣政府, 並了解 台灣的行事曆"},
            ]
    )
    question_message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    reformat_request = HumanMessage(
            content=[
                {"type": "text", "text": "在每一行 只用 日期 和 節日名稱 顯示, 且日期格式為 YYYY-MM-DD. 並用 json format 包裝, json 的第一層是 Result. json format 中 日期 用 date 取代. 節日名稱 用 name 取代."},
            ]
    )

    all_msgs = [ai_message, question_message, reformat_request]

    response = llm.invoke(all_msgs)

    return response.content
    
def generate_hw02(question):
    pass
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response
