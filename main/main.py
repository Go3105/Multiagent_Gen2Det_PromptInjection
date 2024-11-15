from typing import Annotated, Sequence, Literal, TypedDict, Union
import os
import functools
import operator
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_experimental.tools import PythonREPLTool
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from IPython.display import Image, display
from datasets import load_dataset
import json
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "Discuss Gen2Det PromptInjection"

class OverallState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    next: str