import json
import argparse
import os
import random
import time
from tqdm import tqdm
import re
import pandas as pd
import string
import sys

from openai.types import Completion as OpenAICompletion
from openai import RateLimitError as OpenAIRateLimitError
from openai import APIError as OpenAIAPIError
from openai import Timeout as OpenAITimeout

import requests

def call_gpt_4o_mini(prompt):
    url = "YOUR API BASE URL"
    headers = {
        "Authorization": "Bearer YOUR API KEY",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"Error {response.status_code}: {response.text}"


def check_tags_balance(solution_str: str) -> bool:
    """Check whether tags are correctly matched
    
    Args:
        solution_str: The string to be checked
    
    Returns:
        bool: Whether all tags are correctly matched
    """
    # Tag pairs to check
    tags_to_check = ['tool_call', 'think', 'answer']
    
    for tag in tags_to_check:
        # Count the number of start and end tags
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        
        start_count = solution_str.count(start_tag)
        end_count = solution_str.count(end_tag)
        
        # If the number of start and end tags are not equal, return False
        if start_count != end_count:
            return False
            
        # Check nesting order (ensure the end tag does not appear before the start tag)
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
    """Preprocess text for NQ dataset scoring
    
    Processing steps:
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

PROMPT='''You will be given a question and its ground truth answer list where each item can be a ground truth answer. Provided a pred_answer, you need to judge if the pred_answer correctly answers the question based on the ground truth answer list.
You should first give your rationale for the judgement, and then give your judgement result (i.e., correct or incorrect).

Here is the criteria for the judgement:
1. The pred_answer doesn't need to be exactly the same as any of the ground truth answers, but should be semantically same for the question.
2. Each item in the ground truth answer list can be viewed as a ground truth answer for the question, and the pred_answer should be semantically same to at least one of them.

question: {question}
ground truth answers: {gt_answer}
pred_answer: {pred_answer}

The output should in the following json format:

The output should in the following json format:
\'\'\'json
{
\"rationale\": \"your rationale for the judgement, as a text\",
\"judgement\": \"your judgement result, can only be \'correct\' or \'incorrect\'\"
}
\'\'\'
Your output:
'''

def get_json(json_str):
    import json
    import re

    # Use regex to extract the JSON part within curly braces
    try:
        match = re.search(r"\{.*\}", json_str, re.DOTALL)
        if match:
            json_str = match.group()
            data = json.loads(json_str)
            return data
        else:
            return {}
    except:
        return {}

def get_mbe_result(question,gts,pred_answer):
    judgement = ""
    try_cnt = 0
    while True:
        prompt = PROMPT.replace("{question}",question).replace("{gt_answer}",str(gts)).replace("{pred_answer}",pred_answer)
        try:
            batch_responses = call_gpt_4o_mini(prompt)
            judgement = get_json(batch_responses)
            print(judgement)
            if "judgement" in judgement:
                judgement = judgement["judgement"]
            if judgement in ["correct", "incorrect"]:
                if judgement == "correct":
                    return 1.0
                else:
                    return 0.0
        except:
            try_cnt += 1
            if try_cnt > 100:
                return 0.0

def compute_score(question,solution_str, ground_truth, val_type='f1',cot=False) -> float:
    solution_str = solution_str.lower()
    ground_truth = ground_truth.lower()
    ground_truths = ground_truth.split("<|answer_split|>")
    # First check whether the tags are correctly matched (whether the format is correct)
    if cot == True:
        solution_str = solution_str + "</answer>"
    solution_str = solution_str.split("<|im_start|>assistant")[-1]
    if not check_tags_balance(solution_str):
        return -0.0
    # Use regex to extract the content of the first <answer> tag
    try:
        answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            # Preprocess the answer
            answer_content = preprocess_text(answer_content)
        else:
            return -0.0  # If there is no <answer> tag, return -0.0 to indicate format error
    except Exception as e:
        print(f"Error extracting answer content: {e}")
        return -0.0
    
    max_score = 0.0
    
    for gt in ground_truths:
        # Preprocess the ground truth
        gt = preprocess_text(gt)
        
        if val_type == 'em' or val_type == "mbe":
            if gt == answer_content:
                return 1.0
        else:
            # Tokenize the predicted answer and reference answer
            pred_tokens = set(answer_content.split())
            gt_tokens = set(gt.split())
            
            if not gt_tokens:  # Avoid division by zero error
                continue
            if not pred_tokens:
                continue
            
            # Calculate the number of common tokens
            common_tokens = pred_tokens & gt_tokens
            
            # Calculate precision and recall
            precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
            
            # Calculate F1 score
            if precision + recall > 0:  # Avoid division by zero error
                f1 = 2 * (precision * recall) / (precision + recall)
                max_score = max(max_score, f1)
    if val_type == "mbe":
        max_score = get_mbe_result(question,ground_truths,answer_content)
    
    
    return max_score

method = sys.argv[1]
file_path = "../data/test.parquet"
df = pd.read_parquet(file_path)
gts = json.loads(df.to_json(orient="records", force_ascii=False))


with open(f"./{method}_result.json","r",encoding="utf-8") as f:
    answers = json.load(f)
result = {}
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict

result = defaultdict(lambda: {"f1": [], "em": [], "mbe": []})

def compute_metrics(gt, answer, method):
    question = gt["prompt"][0]["content"]
    gt_answer = gt["reward_model"]["ground_truth"]
    data_source = gt["data_source"]
    mbe = 0.0
    if method in ["rag", "cot"]:
        f1 = compute_score(question, answer["response"], gt_answer, "f1", cot=True)
        em = compute_score(question, answer["response"], gt_answer, "em", cot=True)
        mbe = compute_score(question, answer["response"], gt_answer, "mbe", cot=True)
    elif method in ["search_r1_wo_ir","search_r1"]:
        data_source = answer["data_source"]
        question =  answer["question"]
        gt_answer = answer["gt_answer"]
        f1 = compute_score(question, answer["r1_answer"], gt_answer, "f1", cot=False)
        em = compute_score(question, answer["r1_answer"], gt_answer, "em", cot=False)
        mbe = compute_score(question, answer["r1_answer"], gt_answer, "mbe", cot=False)
    elif method in ["r1_searcher"]: 
        data_source = answer["data_source"]
        question =  answer["question"]
        gt_answer = answer["answer"]
        an = f"<answer>{answer['pred_ans']}</answer>"
        f1 = compute_score(question, an, gt_answer, "f1", cot=False)
        em = compute_score(question, an, gt_answer, "em", cot=False)
        mbe = compute_score(question, an, gt_answer, "mbe", cot=False)
    else:
        f1 = compute_score(question, answer["message_str"], gt_answer, "f1", cot=False)
        em = compute_score(question, answer["message_str"], gt_answer, "em", cot=False)
        mbe = compute_score(question, answer["message_str"], gt_answer, "mbe", cot=False)
   
    return data_source, f1, em, mbe

with ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(compute_metrics, gt, answer, method) for gt, answer in zip(gts, answers)]
    
    for future in tqdm(as_completed(futures), total=len(futures)):
        data_source, f1, em, mbe = future.result()
        result[data_source]["f1"].append(f1)
        result[data_source]["em"].append(em)
        result[data_source]["mbe"].append(mbe)

# Average score calculation
for data_source in result:
    result[data_source]["f1"] = sum(result[data_source]["f1"]) / len(result[data_source]["f1"])
    result[data_source]["em"] = sum(result[data_source]["em"]) / len(result[data_source]["em"])
    result[data_source]["mbe"] = sum(result[data_source]["mbe"]) / len(result[data_source]["mbe"])

with open(f"./{method}_score.json","w" ,encoding="utf-8") as f:
    answers = json.dump(result,f,indent=4)

