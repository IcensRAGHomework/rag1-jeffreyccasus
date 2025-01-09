import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)


# HW1 begin
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

import json
import re
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List


class Holiday_Item(BaseModel):
    """Information about a holiday."""
    date: str = Field(..., description="Date of holiday")
    name: str = Field(..., description="Name of holiday.")


class Fianl_Result(BaseModel):
    """Identifying information about all holidays in specific month."""
    Result: List[Holiday_Item]

# Custom parser
def extract_json(message: AIMessage) -> List[dict]:
    """Extracts JSON content from a string where JSON is embedded between \`\`\`json and \`\`\` tags.

    Parameters:
        text (str): The text containing the JSON content.

    Returns:
        list: A list of extracted JSON strings.
    """
    text = message.content
    # Define the regular expression pattern to match JSON blocks
    pattern = r"\`\`\`json(.*?)\`\`\`"

    # Find all non-overlapping matches of the pattern in the string
    matches = re.findall(pattern, text, re.DOTALL)

    # Return the list of matched JSON strings, stripping any leading or trailing whitespace
    try:
        return [json.loads(match.strip()) for match in matches]
    except Exception:
        raise ValueError(f"Failed to parse: {message}")



def generate_hw01(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

    # Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你目前扮演台灣政府, 使用者針對特定年份的台灣行事曆做查詢, 請列出所有符合月份的紀念日."
                "Output your answer as JSON that  "
                "matches the given schema: \`\`\`json\n{schema}\n\`\`\`. "
                "Make sure to wrap the answer in \`\`\`json and \`\`\` tags",
            ),
            ("human", "{query}"),
        ]
    ).partial(schema=Fianl_Result.model_json_schema())

    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=Fianl_Result)

    #print(prompt.format_prompt(query=question).to_string())

    #all_msgs = [ai_message, question_message, reformat_request]
    #response = llm.invoke(all_msgs)

    chain = prompt | llm | parser
    response = chain.invoke({"query": question})

    return json.dumps(response, ensure_ascii = False)

# HW2 begin
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import json
import requests

def get_year(question):
    llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )
    response = llm.invoke([
        SystemMessage(content="你只能回答問題中的年份, 並以數字表示"),
        HumanMessage(content=question),
        #HumanMessage(content="請問是哪個年份?"),
        #SystemMessage(content="請只回答問題中的年份, 並以數字表示"),
    ])
    return response.content

def get_month(question):
    llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )
    response = llm.invoke([
        SystemMessage(content="你只能回答問題中的月份, 並以數字表示"),
        #SystemMessage(content="請只回答問題中的月份, 並以數字表示"),
        HumanMessage(content=question),
        #SystemMessage(content="請只回答問題中的月份, 並以數字表示"),
        #SystemMessage(content="請只回答問題中的月份, 並以數字表示"),
    ])
    return response.content

def generate_hw02(question):
    # Fetch holiday from Calendarific website via webapi and personal api key
    Calendarific_api_key = "JQeWnmY3xqc6y2jRtEhdL58tQY3lKdA5"
    query_year = get_year(question)
    query_month = get_month(question)
    api_url = "https://calendarific.com/api/v2/holidays?&api_key="+Calendarific_api_key+"&country=TW&language=zh&year="+str(query_year)+"&month="+str(query_month)
    webapi_response = requests.get(api_url)
    calendarific_response_json = webapi_response.json()
    #print(calendarific_response_json["date"]["iso"])
    #print(calendarific_response_json["name"])

    # Convert and filter fetch data to json format in homework style
    response_array = []
    for item in calendarific_response_json["response"]["holidays"]:
        item_object = {
                'date': item["date"]["iso"],
                'name': item["name"],
            }
        response_array.append(item_object)
    final_response_json = json.dumps(response_array)
    final_response_json = "{ \"Result\": " + final_response_json + " }"

    """
    holiday_item_formatting = " \"date\": \"{0}\", \"name\": \"{1}\" "

    final_response_json = "{ \"Result\": [ "
    for item in calendarific_response_json["response"]["holidays"]:
        final_response_json = final_response_json + " { "
        final_response_json = final_response_json + holiday_item_formatting.format(str(item["date"]["iso"]), str(item["name"]))

        final_response_json = final_response_json + " } ,"
    final_response_json = final_response_json[:-1]  # remove the last and useless ,
    final_response_json = final_response_json + " ] }"
    """

    return final_response_json

# HW03 begin

from typing import List

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

def generate_hw03(question2, question3):
    # Get response from hw02 (get calendar from Calendarific website)
    feedback_hw02 = generate_hw02(question2)

    # Prepare Open AI
    llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )

    prompt = ChatPromptTemplate.from_messages([
        ("ai", "{holiday_list}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm

    # Begin conversion with history
    #history = get_by_session_id("ask_holiday_session_id")
    #history.add_message(AIMessage(content="hello"))
    #print(store)  # noqa: T201

    chain_with_history = RunnableWithMessageHistory(
        chain,
        # Uses the get_by_session_id function defined in the example
        # above.
        get_by_session_id,
        input_messages_key="question",
        history_messages_key="history",
    )

    result1 = chain_with_history.invoke(
        {"holiday_list": feedback_hw02, 
         "question": question3},
        config={"configurable": {"session_id": "foo"}}
    )
    print(result1.content)

    result_add = chain_with_history.invoke(
        {"holiday_list": feedback_hw02, 
         "question": "這節日如果不在之前的清單, 並且需要被加入, 請回答 true, 反之則回答 false"},
        config={"configurable": {"session_id": "foo"}}
    )
    print(result_add.content)

    result_reason = chain_with_history.invoke(
        {"holiday_list": feedback_hw02, 
         "question": "請用一行, 請解釋一下需要加入或不加入的原因, 並且額外將目前已存在的所有節日, 只列出節日中文名稱"},
        config={"configurable": {"session_id": "foo"}}
    )
    print(result_reason.content)
    remove_specific_char_reason = result_reason.content.replace("\"", "") # remove " inside string
    print(remove_specific_char_reason)

    #prepase final json
    add_or_not_string_formatting = " \"add\": {0},   \"reason\": \"{1}\" "
    final_response = add_or_not_string_formatting.format(result_add.content.lower(), remove_specific_char_reason)
    final_response = " { \"Result\": {  " + final_response + " }  }"

    return final_response

# HW04 begin
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

def generate_hw04(question):
    llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )

    # get image url from github directly
    image_url = "https://raw.githubusercontent.com/IcensRAGHomework/rag1-jeffreyccasus/refs/heads/main/baseball.png?raw=true"

    score_response = llm.invoke([
        SystemMessage(content="只回答分數的數字"),
        HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": question},
            ],
        )
    ])

    #prepase final json
    score_formatting = " \"score\": {0}"
    final_response = score_formatting.format(score_response.content)
    final_response = " { \"Result\": { " + final_response + " }  }"

    return final_response
    
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
