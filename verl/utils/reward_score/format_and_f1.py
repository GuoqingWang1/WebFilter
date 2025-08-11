from openai import OpenAI
import re
import difflib
import string


def check_tags_balance(solution_str: str) -> bool:
    """Check whether tags are properly paired
    
    Args:
        solution_str: The string to be checked
    
    Returns:
        bool: Whether all tags are properly paired
    """
    # Tag pairs to be checked
    tags_to_check = ['tool_call', 'think', 'answer']
    
    for tag in tags_to_check:
        # Count the number of start and end tags
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        
        start_count = solution_str.count(start_tag)
        end_count = solution_str.count(end_tag)
        
        # Return False if the counts of start and end tags do not match
        if start_count != end_count:
            return False
            
        # Check tag nesting order (ensure end tag does not appear before start tag)
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
    
    Processing steps:
    1. Convert to lowercase
    2. Remove punctuation (.,!?;:'"()[]{}...)
    3. Remove extra spaces
    """
    # Replace punctuation with space
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing spaces
    text = text.strip()
    return text



def compute_score(solution_str, ground_truth, val_type='f1') -> float:
    solution_str = solution_str.lower()
    ground_truth = ground_truth.lower()
    ground_truths = ground_truth.split("<|answer_split|>")
    
    # First, check if tags are properly paired (i.e., format is valid)
    if not check_tags_balance(solution_str):
        return -1.0
    
    # Use regex to extract content inside the first <answer> tag
    try:
        answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            # Preprocess the extracted answer
            answer_content = preprocess_text(answer_content)
        else:
            return -1.0  # Return -1.0 if no <answer> tag found (format error)
    except Exception as e:
        print(f"Error extracting answer content: {e}")
        return -1.0
    
    max_score = 0.0
    
    for gt in ground_truths:
        # Preprocess ground truth
        gt = preprocess_text(gt)
        
        if val_type == 'em':
            if gt == answer_content:
                return 1.0
        else:
            # Tokenize predicted and ground truth answers
            pred_tokens = set(answer_content.split())
            gt_tokens = set(gt.split())
            
            if not gt_tokens:  # Avoid division by zero
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
                f1 = 2 * (precision * recall) / (precision + recall)
                max_score = max(max_score, f1)
            
    return max_score
