from openai import OpenAI
import re
import difflib
import string

import random
import json
import logging
import os

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
        
        # If the numbers of start and end tags are not equal, return False
        if start_count != end_count:
            return False
            
        # Check nesting order (ensure the end tag doesn't appear before the start tag)
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
    
    # Strip leading/trailing spaces
    text = text.strip()
    return text

def validate_tool_query_strings(solution_str, allowed_tools=["web_search, local_wiki_search, web_browser"]):
    """
    Validate whether each string in the list matches the `tool:query` format, 
    and tool is in the allowed list
    
    Args:
        strings (list of str): List of strings to validate
        allowed_tools (list of str): List of allowed tool names
    
    Returns:
        list of bool: Whether each string is valid
    """
    # Precompile regex (allow tool name followed by colon, optional spaces around)
    pattern = re.compile(r"^(\w+(?:_\w+)*)\s*:\s*(.+)$")
    allowed_tools_set = set(allowed_tools)  # Convert to set for faster lookup

    if not isinstance(solution_str, str):  # Non-string marked invalid
        return False

    stripped = solution_str.strip()
    match = pattern.match(stripped)
    
    if match:
        tool, query = match.groups()
        if tool in allowed_tools_set and query.strip():  # Ensure query is not empty
            return True
        else:
            return False
    else:
        return False

def compute_score(solution_str, ground_truth, val_type='f1') -> float:
    solution_str = solution_str.lower()
    ground_truth = ground_truth.lower()
    ground_truths = ground_truth.split("<|answer_split|>")
    # First check whether tags are paired correctly (format correctness)
    if not check_tags_balance(solution_str):
        return -1.0
    # Use regex to extract content in the first <answer> tag
    try:
        answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            # Preprocess answer
            answer_content = preprocess_text(answer_content)
        else:
            return -1.0  # If no answer tag, return -1.0 to indicate format error
    except Exception as e:
        print(f"Error extracting answer content: {e}")
        return -1.0
    
    max_score = 0.0
    
    for gt in ground_truths:
        # Preprocess ground truth
        gt = preprocess_text(gt)
        score = 0.0
        if val_type == 'em':
            if gt == answer_content:
                return 1.0
        else:
            # Tokenize predicted answer and reference
            pred_tokens = set(answer_content.split())
            gt_tokens = set(gt.split())
            
            if not gt_tokens:  # Avoid division by zer
                continue
            if not pred_tokens:
                continue
            
            # Count common tokens
            common_tokens = pred_tokens & gt_tokens
            
            # Calculate precision and recall
            precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
            
            # Calculate F1 score
            if precision + recall > 0:  # Avoid division by zero
                score = 2 * (precision * recall) / (precision + recall)
            elif check_keywords_in_toolcalls(solution_str):
                score = 0.1
            else:
                score = 0.0
            max_score = max(max_score, score)


    # save_results_to_file(data_source, prompt_str, solution_str, answer_content, ground_truths, max_score)
    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"------------- [Training] --------------")
        print(f"Solution string: {solution_str}")
        print(f"Extracted answer: {answer_content}")
        print(f"Golden answers: {ground_truths}")
        print(f"score: {max_score}")
            
    return max_score