# =============================================================================
# Based on the Search-R1 example from the Search-R1 project.
#
# Original Authors: Jin Bowen, Zeng Hansi, Yue Zhenrui, Wang Dong, Zamani Hamed, Han Jiawei
#
# License: Apache 2.0
# Project URL: https://github.com/PeterGriffinJin/Search-R1
# =============================================================================

from sympy import SYMPY_DEBUG
import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from scrl.llm_agent.tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
import json
import numpy as np
import time
from datetime import datetime
from time import strftime, gmtime

@dataclass
class GenerationConfig:
    max_turns: int
    num_gpus: int
    data_writing_file: str = None
    signal_writing_file: str = None
    model_name: str = None
    RESPONSE_SIGNAL: int = 0
    QUERY_SIGNAL: int = 1
    n: int = 1,
    project_name: str = None,
    experiment_name: str = None,
    search_engine: str = "rag",
    nnodes: int = 1


TOOLS_FOR_WIKI = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for relevant information. You should use this tool if the historical page content is not enough to answer the question. Or last search result is not relevant to the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "The query to search, which helps answer the question"
                        },
                        "description": "The queries to search"
                    }
                },
                "required": ["query"],
                "minItems": 1,
                "uniqueItems": True
            }
        }
    }
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for relevant information from google. You should use this tool if the historical page content is not enough to answer the question. Or last search result is not relevant to the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",  
                        "items": {    
                            "type": "string",
                            "description": "The query to search, which helps answer the question"
                        },
                        "description": "The queries to search"
                    }
                },
                "required": ["query"],
                "minItems": 1,
                "uniqueItems": True
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browse_webpage",
            "description": "Browse the webpage and return the content that not appeared in the conversation history. You should use this tool if the last action is search and the search result maybe relevant to the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url_list": {"type": "array",
                            "items": {
                                "type": "string",
                                "description": "The chosen url from the search result, do not use url that not appeared in the search result"
                            },
                            "description": "The chosen urls from the search result."
                        },
                },
                "required": ["url_list"]
            }
        }
    }
]


class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id
        ))
        
        if self.config.search_engine == "rag":
            self.tools = TOOLS_FOR_WIKI
            self.system_prompt =  f"""## Background information 
* Today is {strftime("%Y-%m-%d", gmtime())}
* You are Deep AI Research Assistant

The question I give you is a complex question that requires a *deep research* to answer.

I will provide you with one tool to help you answer the question:
* A web search tool to help you perform search.

You don't have to answer the question now, but you should first think about the research plan or what to search next.

Your output format should be one of the following two formats:

<think>
YOUR THINKING PROCESS
</think>
<answer>
YOUR ANSWER AFTER GETTING ENOUGH INFORMATION
</answer>

or

<think>
YOUR THINKING PROCESS
</think>
<tool_call>
YOUR TOOL CALL WITH CORRECT FORMAT
</tool_call>

You should always follow the above two formats strictly.
Only output the final answer (in words, numbers or phrase) inside the <answer></answer> tag, without any explanations or extra information. If this is a yes-or-no question, you should only answer yes or no.
"""
        elif self.config.search_engine == "online_search":
            self.tools = TOOLS
            from scrl.handler.init_prompt import SYSTEM_PROMPT
            self.system_prompt = SYSTEM_PROMPT
        else:
            assert False


    def _update_right_side(self, original_right_side: Dict, 
                           cur_responses: torch.Tensor,
                           next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side of rollings."""
        if next_obs_ids is not None:
            responses = self.tensor_fn.concatenate_with_padding(
                [original_right_side['responses'], cur_responses, next_obs_ids],
                pad_to_left=False
            )
        else:
            responses = self.tensor_fn.concatenate_with_padding(
                [original_right_side['responses'], cur_responses],
                pad_to_left=False
            )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        
        return {'responses': responses[:, :effective_len]}

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, next_obs_ids: torch.Tensor) -> DataProto:
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        return DataProto.from_dict({
                'input_ids': new_input_ids[:, -effective_len:],
                'position_ids': new_position_ids[:, -effective_len:],
                'attention_mask': new_attention_mask[:, -effective_len:]
            })

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']
        return next_obs_ids
        
    def postprocess_predictions(self, rollings_active: DataProto, gen_output: DataProto) -> Tuple[List[int], List[bool]]:
        """Postprocess predictions to remove padding and convert to list of strings."""
        """return: list of query contents including history"""

        pass
        return [{"prompt":""} for _ in range(rollings_active.batch['input_ids'].shape[0])]

    def execute_predictions(self, 
        tool_call_list: List[Tuple[int, str, str, str]], total_number: int = 4096
    ) :
        query_contents = [{"idx": tool_call[0], "question": tool_call[1], "think": tool_call[2], "tool_call": tool_call[3], "total_number":total_number} for tool_call in tool_call_list]
        with open(self.config.signal_writing_file, 'r') as f:
            signal_contents = json.load(f)
        assert signal_contents['signal'] == self.config.RESPONSE_SIGNAL
        with open(self.config.data_writing_file, 'w', encoding='utf-8') as f:
            json.dump(query_contents, f, indent=4, ensure_ascii=False)
        with open(self.config.data_writing_file, 'r', encoding='utf-8') as f:
            query_contents_check = json.load(f)
        assert query_contents == query_contents_check
        with open(self.config.signal_writing_file, 'w', encoding='utf-8') as f:
            json.dump({'signal': self.config.QUERY_SIGNAL}, f, indent=4, ensure_ascii=False)
        response_finish = False
        while not response_finish:
            with open(self.config.signal_writing_file, 'r', encoding='utf-8') as f:
                signal_contents = json.load(f)
            if signal_contents['signal'] == self.config.RESPONSE_SIGNAL:
                response_finish = True
            else:
                time.sleep(10)
        with open(self.config.data_writing_file, 'r', encoding='utf-8') as f:
            query_contents = json.load(f)
        return query_contents

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus * self.config.nnodes
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def parse_question(self, input_ids: torch.Tensor) -> str:
        """Parse question to get the query content."""
        query_contents = self.tokenizer.batch_decode(input_ids)
        query_contents = [re.sub(r'^(<\|endoftext\|>)+', '', content) for content in query_contents]
        query_contents = [content.split("<|im_start|>user\n")[1].split("<|im_end|>")[0] for content in query_contents]
        return query_contents

    def parse_response(self, input_ids: torch.Tensor, think: bool = False) -> List[Tuple[bool, str, str]]:
        """Parse response to get the thinking process and answer or tool call.
            return: [(is_stop, thinking, answer/tool_call), ...]
        """
        response_contents = self.tokenizer.batch_decode(input_ids)
        results = []
        for i, content in enumerate(response_contents):
            if think:
                content = "<think>" + content
            if "<think>" in content and "<answer>" in content:
                if "</think>" not in content or "</answer>" not in content:
                    results.append((True, "", ""))
                else:
                    think = content.split("<think>")[1].split("</think>")[0]
                    answer = content.split("<answer>")[1].split("</answer>")[0]
                    results.append((True, think, answer))
            elif "<think>" in content and "<tool_call>" in content:
                if "</tool_call>" not in content or "</think>" not in content:
                    results.append((True, "", ""))
                else:
                    think = content.split("<think>")[1].split("</think>")[0]
                    tool_call = content.split("<tool_call>")[1].split("</tool_call>")[0]
                    try:
                        tool_call = json.loads(tool_call)
                        assert "name" in tool_call, "no vliad function name in tool_call"
                        assert "arguments" in tool_call, "no valid arguments in tool_call"
                        assert tool_call["name"] in ["web_search", "browse_webpage"], "invalid tool name"
                        if tool_call["name"] == "web_search":
                            assert "query" in tool_call["arguments"], "no valid query in tool_call"
                            assert isinstance(tool_call["arguments"]["query"], list), "query should be a list"
                        elif tool_call["name"] == "browse_webpage":
                            assert "url_list" in tool_call["arguments"], "no valid url_list in tool_call"
                            assert isinstance(tool_call["arguments"]["url_list"], list), "url_list should be a list"
                            assert len(tool_call["arguments"]["url_list"]) >= 1, "url_list number must be greater than 0"
                        results.append((False, think, tool_call))
                    except Exception as e:
                        print(f"model tool call format error: {e}")
                        results.append((True, "", ""))
            else:
                results.append((True, "", ""))
        return results

    def run_llm_loop(self, gen_batch: DataProto, global_steps: int) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        node_rank = int(os.environ["PET_NODE_RANK"])
        print(f"node {node_rank} gains {len(gen_batch.batch['input_ids'])} datas!",flush=True)
        query_contents = self.parse_question(gen_batch.batch['input_ids'])
        messages_list = []
        agent_grpo_idx = []
        for idx, query_content in enumerate(query_contents):
            for _ in range(self.config.n):
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query_content}
                ]
                messages_list.append(messages)
                agent_grpo_idx.append(idx)
        activate_list = [i for i in range(len(messages_list))]
        message_string_list = ["" for _ in range(len(messages_list))]
        
        output_dir = f"./outputs/{self.config.project_name}/{self.config.experiment_name}/rollout"
        if not os.path.exists(output_dir):
            print(f"Directory not exist, create at {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        for step in range(self.config.max_turns):
            print(f"node {node_rank} step {step} start!")
            activate_messages_list = [messages_list[i] for i in activate_list]
            if activate_list == []:
                break

            rollings_active = self.tokenizer.apply_chat_template(activate_messages_list, add_generation_prompt=True, tools=self.tools, tokenize=False)
            think = True
            
            if think:
                rollings_active = [rolling + "<think>" for rolling in rollings_active]
            else:
                rollings_active = [rolling for rolling in rollings_active]
            rollings_active = self.tokenizer(rollings_active, return_tensors="pt",padding=True)

            pad_mask = rollings_active['input_ids'] != self.tokenizer.pad_token_id
            sorted_indices = pad_mask.to(torch.int64).argsort(dim=1, stable=True)
            rollings_active['input_ids'] = rollings_active['input_ids'].gather(1, sorted_indices)
            rollings_active['attention_mask'] = rollings_active['attention_mask'].gather(1, sorted_indices)
            
            attention_mask = rollings_active['attention_mask']
            rollings_active['position_ids'] = self.tensor_fn.create_position_ids(attention_mask)
            
            with open(f"./outputs/{self.config.project_name}/{self.config.experiment_name}/rollout/rollout_step_{global_steps}_round_{step}.json", "w", encoding='utf-8') as f:
                step_write_list = []
                for i, input_ids in enumerate(rollings_active['input_ids']):
                    step_write_list.append({
                        "idx": activate_list[i],
                        "question": query_contents[agent_grpo_idx[activate_list[i]]],
                        "input_ids_no_pad": self.tokenizer.decode(input_ids, skip_special_tokens=False).replace("<|endoftext|>", ""),
                    })
                json.dump(step_write_list, f, indent=4, ensure_ascii=False)
            print(f"rollout_step_{global_steps}_round_{step}.json done")
            
            print(f"node {node_rank}, turn {step} rollings_active is {len(rollings_active['input_ids'])} datas")
            rollings_active = DataProto.from_dict({
                'input_ids': rollings_active['input_ids'],
                'attention_mask': rollings_active['attention_mask'],
                'position_ids': rollings_active['position_ids'],
            })
            
            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info
            print(f"node {node_rank}, turn {step} gen_output {len(gen_output.batch['responses'])} datas")

            results = self.parse_response(gen_output.batch['responses'], think=think)
            assert len(results) == len(activate_list) 
            activate_list_copy = []
            tool_call_list = []
            for i in range(len(results)):
                if results[i][0]:
                    message_string_list[activate_list[i]] = self.tokenizer.decode(rollings_active.batch['input_ids'][i], skip_special_tokens=False).replace("<|endoftext|>", "") + self.tokenizer.decode(gen_output.batch['responses'][i], skip_special_tokens=False).replace("<|endoftext|>", "")
                else:
                    activate_list_copy.append(activate_list[i])
                    tool_call_list.append((activate_list[i], messages_list[activate_list[i]][1]["content"], results[i][1], results[i][2]))
                    
            tool_call_list = self.execute_predictions(tool_call_list,len(messages_list))
            print(f"node {node_rank}, turn {step} tool_call_list {len(tool_call_list)} datas")
            for i in range(len(tool_call_list)):
                messages_list[tool_call_list[i]['idx']].append(
                    {
                        "role": "assistant", 
                        "content": "<think>" + tool_call_list[i]['think'] + "</think>", 
                        "tool_calls": [
                                        {
                                            "type": "function", 
                                            "function": tool_call_list[i]['tool_call']
                                        }
                                    ]
                    }
                )
                messages_list[tool_call_list[i]['idx']].append(
                    {
                        "role": "tool", 
                        "name": tool_call_list[i]['tool_call']["name"],
                        "content": tool_call_list[i]['content']
                    }
                )
            print(f"Round {step} finished. Node {node_rank} originally had {len(activate_list)} queries, now has {len(activate_list_copy)} queries.")

            activate_list = activate_list_copy
        if activate_list != []:
            for i in activate_list:
                message_string_list[i] = self.tokenizer.apply_chat_template(messages_list[i], add_generation_prompt=True, tools=self.tools, tokenize=False)
        
        response_str_list = []
        initial_prompt_list = []
        for i, messages in enumerate(messages_list):
            initial_prompt = self.tokenizer.apply_chat_template(messages[0:2], add_generation_prompt=True, tools=self.tools, tokenize=False)
            initial_prompt_list.append(initial_prompt)
            response_str_list.append(message_string_list[i][len(initial_prompt):])
        
        prompts_tokenizered = self.tokenizer(initial_prompt_list, return_tensors="pt",padding=True)

        prompts_repeated = prompts_tokenizered['input_ids']
        pad_mask = prompts_repeated != self.tokenizer.pad_token_id
        sorted_indices = pad_mask.to(torch.int64).argsort(dim=1, stable=True)

        prompts_repeated = prompts_repeated.gather(1, sorted_indices)
        prompts_attention_mask = prompts_tokenizered['attention_mask'].gather(1, sorted_indices)

        responses = self.tokenizer(response_str_list, return_tensors="pt",padding=True)['input_ids']
        
        responses_attention_mask = self.tokenizer(response_str_list, return_tensors="pt",padding=True)['attention_mask']
        attention_mask = torch.cat((prompts_attention_mask, responses_attention_mask), dim=-1)
        position_ids = self.tensor_fn.create_position_ids(attention_mask)
        
        message_tensor = DataProto.from_dict({
            'prompts': prompts_repeated,
            'responses': responses,
            'input_ids': torch.cat((prompts_repeated, responses), dim=-1),
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        })
        message_tensor.meta_info.update(meta_info)
        message_tensor.non_tensor_batch['agent_grpo_idx'] = np.array(agent_grpo_idx, dtype=object)
        print("generation done")
        
        with open(f"./outputs/{self.config.project_name}/{self.config.experiment_name}/rollout/rollout_step_{global_steps}.json", "w", encoding='utf-8') as f:
            write_list = []
            for i, message_str in enumerate(message_string_list):
                write_list.append({
                    "idx": i,
                    "question": query_contents[agent_grpo_idx[i]],
                    "message_str": message_str
                })
            json.dump(write_list, f, indent=4, ensure_ascii=False)
            print(f"rollout_step_{global_steps}.json done")
        print(f"node {node_rank} message_string_list {len(message_string_list)}")
              
        return message_string_list, message_tensor
    
