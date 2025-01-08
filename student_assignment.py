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

    return response
    
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
