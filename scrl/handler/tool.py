import time
import json
import threading
import yaml
from typing import List, Dict
from smolagents import Tool
from openai import OpenAI
from loguru import logger
import traceback


class LocalWikiSearchTool(Tool):
    name = "local_wiki_search"
    description = "This is a wiki retriever, which can retrieve relevant information snippets from wiki through keywords."
    inputs = {
        "query": {
            "type": "string",
            "description": "Can query specific keywords or topics to retrieve accurate and comprehensive information. The query is preferably English keywords.",
        }
    }
    example = {"name": "local_wiki_search", "arguments": {"query": "xxxx"}}
    output_type = "string"

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

    def forward(self, query:str):
        pass


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web for relevant information from google. You should use this tool if the historical page content is not enough to answer the question. Or last search result is not relevant to the question."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to search, which helps answer the question",
        }
    }
    example = {"name": "web_search", "arguments": {"query": "xxxx"}}
    output_type = "object"

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.search_cache_lock = threading.Lock()

    def forward(self, query:str):
        pass


class BrowseWebpageTool(Tool):
    name = "browse_webpage"
    description = "Browse the webpage and return the content that not appeared in the conversation history. You should use this tool if the last action is search and the search result maybe relevant to the question."
    inputs = {
        "url_list": {
            "type": "array",
            "description": "The chosen urls from the search result, do not use url that not appeared in the search result",
        },
        "query": {
            "type": "string",
            "description": "These queries aim to retrieve information from the URL webpage.",
        }
    }
    example = {"name": "browse_webpage", "arguments": {"url_list": ["http://www.dada.com", "xxxx"], "query": "xxxx"}}
    output_type = "array"

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from tools_server.webbrower.read_agent import ReadingAgent
        self.config = config
        self.read_agent = ReadingAgent(self.config, client = OpenAI(
            api_key=self.config['openai_api_key'], # To Get Free API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
            base_url=self.config["openai_base_url"],
        ))
        self.browse_cache_lock = threading.Lock()
        # self.browse_cache = load_tool_cache_data(self.config, '/cache/browse_cache.json')
        self.browse_cache = {}

    def forward(self, url_list: List[str], query: str):
        pass




def get_tools(config):
    available_tools = config['available_tools']
    tool_classes = {
        "local_wiki_search": LocalWikiSearchTool,
        "web_search": WebSearchTool,
        "browse_webpage": BrowseWebpageTool
    }
    AVAILABLE_TOOLS = {
        name: tool_classes[name]
        for name in available_tools if name in tool_classes
    }
    return AVAILABLE_TOOLS

    
