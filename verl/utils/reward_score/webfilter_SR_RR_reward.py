# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import requests
import json
from multiprocessing import Pool
from functools import partial
from openai import OpenAI
import re
import difflib
import string
import os
import traceback

import random
from itertools import islice

import logging
logging.basicConfig(level=logging.INFO)

def extract_question(text):
    # Use regex to match the content between the last user and assistant
    pattern = r'user\s*(.*)'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return ""
        
def save_results_to_file(data_source, prompt_str, solution_str, answer_content, ground_truths, max_score):
    """Save results to a file with a counter"""
    # The generation part has a counter, read it here
    try:
        # Create output dir
        os.makedirs(os.path.dirname("./outputs/eval/"), exist_ok=True)
        
        count_file_path = "./outputs/eval/count.txt"
        current_count = 0
        
        # Process counter file: if exists, read it
        if os.path.exists(count_file_path):
            with open(count_file_path, 'r', encoding='utf-8') as f:
                count_str = f.read().strip()
                if count_str.isdigit():
                    current_count = int(count_str)
        else:
            # Create a new counter file
            with open(count_file_path, 'w', encoding='utf-8') as f:
                f.write('0')
                logging.info("Create a new counter file: count.txt")
        
        # Generate a filename with the counter
        json_file_path = f"./outputs/eval/train_{current_count}.jsonl"
        
        question = extract_question(prompt_str)
        
        # Save data to the new file
        save_json = {
            "question": question,
            "answer_content": answer_content,
            "ground_truths": ground_truths,
            "score": max_score,
            "data_source":data_source,
            "solution_str": solution_str,
        }
        json_line = json.dumps(save_json, ensure_ascii=False)
        
        # Write to the JSONL file
        with open(json_file_path, 'a', encoding='utf-8') as f:
            f.write(json_line + '\n')
            logging.info(f"File saved: {json_file_path}")

            
    except (IOError, OSError) as e:
        logging.error(f"File write failed: {e}")
    except Exception as e:
        logging.error(f"Unknown error: {e}")



def call_api(record, model_name, prompt_name='prompt',
             api_url="http://localhost:8000/v1/chat/comptions",
             use_new_format=False, headers=None,
             n=1, stop_token=None, top_p=0.9, top_k=40, temperature=0.7):
    """
    Process a single record and call the API using the format specified by the use_new_format parameter.

    Args:
      record (dict): A single record (e.g., containing a "prompt" field)
      model_name (str): Model name, used as a parameter in the payload
      prompt_name (str): Key name in the record used to find the user input, default 'prompt'
      api_url (str): API URL address
      use_new_format (bool): Whether to use the new format to call the API, default False
      headers (dict): Used only when use_new_format is True, specifies HTTP request headers
      n (int): Number of generations, default 1
      stop_token (str or list): Stop token(s) for generation, default None
      top_p (float): Nucleus sampling parameter, default 0.9
      top_k (int): Top-k sampling parameter, default 40
      temperature (float): Temperature parameter, default 0.7

    Returns:
      dict: Contains the input prompt and API return results, or error information.
    """
    prompt = record.get(prompt_name, "")
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        # "max_length": 8192,  
        "top_p": top_p,
        "top_k": top_k,
        "temperature": temperature,
        "n": n,
        "stop": stop_token
    }
    try:
        response = requests.post(api_url, json=payload, timeout=60*60)
        response.raise_for_status()  # Non-200 status codes will raise an exception
        result = response.json()
        return {"prompt": prompt, "payload": payload, "result": result}
    except Exception as e:
        return {"prompt": prompt, "payload": payload, "error": str(e)}


def get_model_response(prompt):
    model_name = "auto"  # Replace according to actual situation
    api_url = "http://33.212.71.243:8000/v1/chat/completions"
    use_new_format = False

    # Additional parameters that can be adjusted as needed
    n = 1
    stop_token = None
    top_p = 0.95
    top_k = 50
    temperature = 0.8
    inputs = {}
    inp = prompt
    inputs['prompts'] = inp
    headers_new = {'Content-Type': 'application/json'}

    results = call_api(
        inputs,
        model_name,
        prompt_name="prompts",
        api_url=api_url,
        use_new_format=use_new_format,
        headers=headers_new,
        n=n,
        stop_token=stop_token,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature
    )

    try:
        output = results['result']['choices'][0]['message']['content']
    except Exception as e:
        # Handle exceptions: can return a default value or raise an exception
        output = f"Error: Unable to fetch response due to {str(e)}"

    return output



def WebFilter_eval_prompt(process_str, result_str, reference_str):
    prompt = f'''You are an expert in the evaluation of research methodologies. Your task is to assess the quality of the research process and outcome generated by the Deep-Research system. The evaluation should be conducted with reference to a gold-standard answer, focusing on the two dimensions outlined below.

## Evaluation Criteria

1. **Search Strategy Effectiveness** (`Search_score`) — *Score range: 0.0 to 1.0*
   - **1.0**: Demonstrates highly efficient use of advanced search operators (e.g., `OR`, `AND`, `site:`, `-`, `date:`) to locate precise information with minimal iterations.
   - **0.5**: Basic search strategy is valid and produces relevant results, but lacks optimization or efficiency.
   - **0.0**: No meaningful use of filtering or search operators; the strategy is inefficient or irrelevant.

2. **Result Accuracy** (`llm_judge_score`) — *Score range: 0.0 to 1.0*
   - **1.0**: Fully consistent with the reference answer, including key conclusions and reasoning.
   - **0.5**: Partially correct; some key elements are aligned, but others are incomplete, missing, or require further validation based on the retrieved evidence.
   - **0.0**: The output is incorrect, unrelated to the question, or logically inconsistent with the reference.

## Evaluation Input

- **Research Process Trace**:  
  `{process_str}`

- **Generated Output**:  
  `{result_str}`

- **Reference Answer (Gold Standard)**:  
  `{reference_str}`

## Output Format Requirements

- The evaluation output **must be in strict JSON format**.
- The JSON object must include the following fields:
  - `"Search_score"` (float): Score for search strategy effectiveness.
  - `"llm_judge_score"` (float): Score for final answer accuracy.
  - `"analysis"` (string, in English, ≤100 characters): Brief explanation of any observed deficiencies or reasoning flaws.

## Example Evaluation Output

The search strategy shows limited use of filtering mechanisms (0.3), and the conclusion contains dosage-related inaccuracies (0.0).  
**Analysis**: 1) Lacks temporal filtering 2) Does not verify the applicability of key parameters.

Final output format:
\\boxed{{
  "llm_judge_score": 0.0,
  "Search_score": 0.3,
  "analysis": "Concise analysis in English..."
}}'''
    return prompt



def check_keywords_in_toolcalls(text, keyword_list=["site:", "after:", "before:", " AND ", " OR ", " -", " NOT "]):
    """
    Check whether all query fields in <tool_call> tags contain specified keywords (case-sensitive)
    New features:
    1. When matching "-", it must have a space before it and at least one character after it
    2. Add match for double-quoted content (e.g., "example")

    Args:
        text: Original text containing tool_call
        keyword_list: List of keywords to check (case-sensitive)

    Returns:
        bool: Whether any keyword is found
    """
    # Match the complete JSON block between <tool_call> and </tool_call> (supports multi-line)
    pattern = r'<tool_call>\s*({[\s\S]*?})\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)

    for i, json_str in enumerate(matches):
        try:
            # Clean JSON string (handle extra spaces/newlines)
            cleaned_json = json_str.strip()
            # Parse JSON object
            data = json.loads(cleaned_json)

            # Extract query field (multi-level safety)
            query = ""
            if isinstance(data, dict):
                query = data.get('arguments', {}).get('query', '')

            # Check if any keyword exists (case-sensitive)
            for keyword in keyword_list:
                if keyword == " -":
                    # Special handling: match space before minus, minus sign, and at least one non-space char after
                    if re.search(r'\s-\S+', query):
                        print(f"Found keyword '-' in query: {query}")
                        return True
                else:
                    # Escape special chars and build regex (match complete word boundary)
                    escaped_keyword = re.escape(keyword)
                    regex_pattern = rf'\b{escaped_keyword}\b'
                    if re.search(regex_pattern, query):  
                        print(f"Found keyword '{keyword}' in query: {query}")
                        return True

            # Check for double-quoted content (at least one character)
            if re.search(r'"[^"]+"', query):  
                print(f"Found double-quoted content in query: {query}")
                return True

        except json.JSONDecodeError as e:
            print(f"[Error] Failed to parse JSON at #{i+1}: {str(e)}")
            continue
        except Exception as e:
            print(f"[Warning] Error processing entry #{i+1}: {str(e)}")
            continue

    return False

        
def extract_json_from_text(text):
    data = ''  # Using empty string instead of "none" is more standard
    
    try:
        if not isinstance(text, str):
            raise TypeError("Input must be of string type")

        pattern = re.compile(r'\{.*?\}', re.DOTALL)
        matches = pattern.finditer(text)
        
        last_match = None
        for match in matches:
            last_match = match

        if last_match:
            json_str = last_match.group()
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")
        else:
            print("No JSON data found")
            
    except TypeError as te:
        print(f"Type error: {te}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return data





def format_reward(predict_str: str) -> float:
    pattern = re.compile(
        r'<think>.*?</think>\s*'
        r'(<answer>.*?</answer>|<tool_call>.*?</tool_call>)',
        re.DOTALL
    )
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else -1.0




def check_tags_balance(solution_str: str) -> bool:
    """Check whether tags are correctly paired
    
    Args:
        solution_str: The string to check
    
    Returns:
        bool: Whether all tags are correctly paired
    """
    # Tag pairs to check
    tags_to_check = ['tool_call', 'think', 'answer']
    
    for tag in tags_to_check:
        # Count the number of start and end tags
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        
        start_count = solution_str.count(start_tag)
        end_count = solution_str.count(end_tag)
        
        # If the counts of start and end tags are not equal, return False
        if start_count != end_count:
            return False
            
        # Check the nesting order (ensure the end tag does not appear before the start
        last_pos = -1
        while True:
            start_pos = solution_str.find(start_tag, last_pos + 1)
            if start_pos == -1:
                break
                
            end_pos = solution_str.find(end_tag, start_pos)
            if end_pos == -1:
                return False
                
            last_pos = end_pos
            
    return True



def preprocess_text(text: str) -> str:
    """Preprocess text for dataset scoring
    
    Steps:
    1. Convert to lowercase
    2. Remove punctuation (.,!?;:'"()[]{}...)
    3. Remove extra spaces
    """
    # Replace punctuation with spaces
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing spaces
    text = text.strip()
    return text




def compute_score(data_source, prompt_str, solution_str, ground_truth, val_type='f1') -> float:
    solution_str = solution_str.lower()
    ground_truth = ground_truth.lower()
    ground_truths = ground_truth.split("<|answer_split|>")
    if not check_tags_balance(solution_str):
        return -1
    try:
        answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            answer_content = preprocess_text(answer_content)
        else:
            return -1.0
    except Exception as e:
        print(f"Error extracting answer content: {e}")
        return -1.0
    
    try:
        answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            result_str = preprocess_text(answer_content)
        else:
            result_str = ''
    except Exception as e:
        result_str = ''

    ground_truths = ground_truth.split("<|answer_split|>")
    if  len(ground_truths)>1:
        ground_truth = ' and '.join(ground_truths)
        
    prompts = WebFilter_eval_prompt(solution_str,result_str,ground_truth)
    eval_response = get_model_response(prompts)

    json_score = extract_json_from_text(eval_response)
    Search_score = 0.0 
    llm_judge_score = 0.0
    try:
        Search_score = float(json_score['Search_score'])
        llm_judge_score = float(json_score['llm_judge_score'])
    except:
        print(f"parse socre error is {json_score}")

    f1_score = 0.0
    max_score = 0.0
    for gt in ground_truths:
        gt = preprocess_text(gt)
        score = 0.0
        if val_type == 'em':
            if gt == answer_content:
                return 1.0
        else:
            pred_tokens = set(answer_content.split())
            gt_tokens = set(gt.split())
            
            if not gt_tokens:  
                continue
            if not pred_tokens:
                continue
            
            common_tokens = pred_tokens & gt_tokens
            
            precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
            
            if precision + recall > 0:  
                f1_score = 2 * (precision * recall) / (precision + recall)
                score = 0.4 * f1_score + 0.4 * llm_judge_score + 0.2 * Search_score
            elif llm_judge_score > 0:
                score = 0.4 * llm_judge_score + 0.2 * Search_score
            elif check_keywords_in_toolcalls(solution_str):
                score = 0.1
            else:
                score = 0.0
            max_score = max(max_score, score)

    # save_results_to_file(data_source, prompt_str, solution_str, answer_content, ground_truths, max_score, f1_score, llm_judge_score, Search_score)
    do_print = random.randint(1, 60) == 1
    if do_print:
        print(f"------------- [Training] --------------")
        print(f"Solution string: {solution_str}")
        print(f"Extracted answer: {answer_content}")
        print(f"Golden answers: {ground_truths}")
        print(f"score = {max_score} = 0.4 x「f1:{f1_score}」 + 0.4 x「llm_judge_score:{llm_judge_score}」 + 0.2 x「Search_score:{Search_score}」")
        print(f"data_source: {data_source}")

    return max_score


def compute_score_batch(prompts_strs: list, predict_strs: list, ground_truths: list, data_sources: list, default_batch_size=32) -> list:
    """
    Use multiprocessing to compute batch scores in parallel.

    Args:
        predict_strs (list): List of prediction strings
        ground_truths (list): List of reference answer strings
        default_batch_size (int): Default number of items processed per process

    Returns:
        list: A list of scores corresponding to each prediction string
    """
    if len(predict_strs) != len(ground_truths):
        raise ValueError("The number of prediction strings and reference answers is inconsistent!")

    batch_size = min(default_batch_size, len(predict_strs))

    # Package input data as a list of tuples
    inputs = list(zip(prompts_strs, predict_strs, ground_truths, data_sources))
    
    # Define multiprocessing pool
    with Pool(processes=batch_size) as pool:
        # map function distributes tasks to multiple processes
        scores = pool.starmap(compute_score, inputs)

    return scores
