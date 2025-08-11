import yaml
from time import strftime, gmtime
from jinja2 import StrictUndefined, Template
from typing import Dict, Any

from scrl.handler.tools import get_tools

def populate_template(template: str, variables: Dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")

def initialize_system_prompt(system_prompt_templates, tools, today) -> str:
    system_prompt = populate_template(
        system_prompt_templates,
        variables={
            "tools": tools,
            "today": today
        },
    )
    return system_prompt


try:
    config_path = "./scrl/handler/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    SYSTEM_PROMPT = initialize_system_prompt(config['system_prompt'], get_tools(config), strftime("%Y-%m-%d", gmtime()))
    print("SYSTEM_PROMPT",SYSTEM_PROMPT)
except Exception as e:
    print(f"initialize system prompts error: {e}")
    SYSTEM_PROMPT = None
