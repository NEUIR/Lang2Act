"""
GRPO Reward Function (SiliconFlow Adapted)

- answer_score: Evaluated by LLM Judge (Qwen2.5-72B), weight 0.8
- format_score: Check if <think>/<description>/<answer> are complete, weight 0.2
- Requires environment variable: SILICONFLOW_API_KEY
"""

import re
import os
import json
import logging
import time
from typing import Dict, List, Any, Optional

try:
    import openai
except ImportError:
    import subprocess, sys
    print("openai library not found. Installing it now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    import openai

# ==============================================================================
# --- Configure Logging ---
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("RewardFunc_SiliconFlow")

# ==============================================================================
# --- Global Config & Client Init ---
# ==============================================================================
# SiliconFlow Config
API_KEY = os.environ.get("SILICONFLOW_API_KEY")
API_BASE_URL = "https://api.siliconflow.cn/v1/"
LLM_JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct"  # Recommended strong model for evaluation

if not API_KEY:
    log.error("CRITICAL: SILICONFLOW_API_KEY environment variable is missing!")
    raise ValueError("SILICONFLOW_API_KEY is required to run this reward function.")

try:
    client = openai.OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    log.info(f"✅ OpenAI Client initialized for SiliconFlow. Model: {LLM_JUDGE_MODEL}")
except Exception as e:
    log.error(f"Failed to initialize client: {e}")
    raise e

# ==============================================================================
# --- LLM Judge Prompt ---
# ==============================================================================
LLM_JUDGE_SYSTEM_PROMPT = """You are an automated evaluation system. Your ONLY task is to compare a 'Generated Answer' to a 'Reference Answer' and determine if they are equivalent.
Follow these rules STRICTLY:
1.  For EACH item you evaluate, you MUST output EXACTLY ONE `<judge>` tag.
2.  Inside the tag, write 'True' if the generated answer is correct or equivalent to the reference.
3.  Inside the tag, write 'False' if the generated answer is incorrect, incomplete, or different from the reference.
4.  Your entire response MUST consist ONLY of a series of `<judge>` tags, one for each item.
5.  ABSOLUTELY DO NOT provide any text, explanation, reasoning, markdown, or any characters outside of the `<judge>True</judge>` or `<judge>False</judge>` tags.
Example for 2 items:
<judge>True</judge>
<judge>False</judge>
"""

def get_llm_answer_scores(prompts_for_eval: List[Dict]) -> List[float]:
    """Call SiliconFlow API for batch evaluation"""
    if not prompts_for_eval:
        return []

    # Batch configuration
    LLM_JUDGE_BATCH_SIZE = 5  # Adjust based on API limits
    MAX_RETRIES = 3
    RETRY_DELAY = 2

    all_scores: List[float] = []
    
    for i in range(0, len(prompts_for_eval), LLM_JUDGE_BATCH_SIZE):
        chunk = prompts_for_eval[i:i + LLM_JUDGE_BATCH_SIZE]
        log.info(f"Processing chunk {i // LLM_JUDGE_BATCH_SIZE + 1} ({len(chunk)} items)...")

        # Construct User Prompt
        user_message_parts = ["Evaluate the following items and output one <judge> tag per item.\n"]
        for j, prompt in enumerate(chunk):
            item_text = (
                f"\n--- ITEM {j + 1} ---\n"
                f"## Query\n{prompt.get('query', 'N/A')}\n\n"
                f"## Reference Answer\n{prompt.get('reference_answer', 'N/A')}\n\n"
                f"## Generated Answer\n{prompt.get('generated_answer', 'N/A')}"
            )
            user_message_parts.append(item_text)
        user_message = "".join(user_message_parts)

        chunk_scores = []
        
        # Retry loop
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=LLM_JUDGE_MODEL,
                    messages=[
                        {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    max_tokens=max(512, len(chunk) * 20),
                    temperature=0.0,
                )
                
                full_text = response.choices[0].message.content or ""
                judges = re.findall(r'<judge>(True|False|true|false)</judge>', full_text)

                if len(judges) == len(chunk):
                    chunk_scores = [1.0 if j.lower() == 'true' else 0.0 for j in judges]
                    break  # Success, break retry loop
                else:
                    log.warning(f"Mismatch: Expected {len(chunk)} tags, got {len(judges)}. Text: {full_text[:100]}...")
            
            except Exception as e:
                log.warning(f"API Attempt {attempt + 1} failed: {e}")
            
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt)) # Exponential backoff

        # If final attempt fails, default to 0
        if not chunk_scores:
            log.error(f"Failed to score chunk after {MAX_RETRIES} attempts. Defaulting to 0.0")
            chunk_scores = [0.0] * len(chunk)
            
        all_scores.extend(chunk_scores)

    return all_scores

# ==============================================================================
# --- Parsing Logic ---
# ==============================================================================
def parse_generation(response: str) -> Dict[str, Optional[str]]:
    """Parse XML structure"""
    think = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
    description = re.search(r'<description>(.*?)</description>', response, re.DOTALL | re.IGNORECASE)
    answer = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    
    return {
        "think": think.group(1).strip() if think else None,
        "description": description.group(1).strip() if description else None,
        "answer": answer.group(1).strip() if answer else None,
    }

# ==============================================================================
# --- Main Entry: GRPO Reward Calculation ---
# ==============================================================================
def compute_score(reward_inputs: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    Main function called by GRPO training
    Args:
        reward_inputs: List containing prompt, ground_truth, response
    """
    if not reward_inputs:
        return []

    log.info(f"--- compute_score called for {len(reward_inputs)} items ---")

    W_ANSWER = 0.8  # Accuracy weight
    W_FORMAT = 0.2  # Format weight

    parsed_items = [parse_generation(item.get("response", "")) for item in reward_inputs]

    # 1. Compute format score
    format_scores = [
        1.0 if all(p.get(k) for k in ["think", "description", "answer"]) else 0.0
        for p in parsed_items
    ]

    # 2. Prepare Judge data
    prompts_for_eval = []
    mapping_indices = []  # Track indices needing Judge

    answer_scores = [0.0] * len(reward_inputs)

    for i, (item, p) in enumerate(zip(reward_inputs, parsed_items)):
        gt = (item.get("ground_truth") or "").strip()
        ans = (p.get("answer") or "").strip()
        
        # Only call API if answer extracted and GT exists
        if gt and ans:
            prompts_for_eval.append({
                "query": item.get("prompt", ""),
                "reference_answer": gt,
                "generated_answer": ans,
            })
            mapping_indices.append(i)

    # 3. Batch API call
    if prompts_for_eval:
        llm_scores = get_llm_answer_scores(prompts_for_eval)
        for score, idx in zip(llm_scores, mapping_indices):
            answer_scores[idx] = score

    # 4. Synthesize final scores
    final_scores = []
    for i in range(len(reward_inputs)):
        overall = (W_ANSWER * answer_scores[i]) + (W_FORMAT * format_scores[i])
        
        record = {
            "overall": round(overall, 6),
            "format_score": format_scores[i],
            "answer_score": answer_scores[i]
        }
        final_scores.append(record)

    # Log the first sample
    if final_scores:
         log.info(f"Sample 0 Score: {final_scores[0]} | GT: {reward_inputs[0].get('ground_truth')} | Pred: {parsed_items[0].get('answer')}")

    return final_scores

# ==============================================================================
# --- Quick Test ---
# ==============================================================================
if __name__ == "__main__":
    # Requires export SILICONFLOW_API_KEY="sk-..."
    print("--- Testing Reward Function with SiliconFlow ---")
    
    test_data = [{
        "prompt": "1+1=?",
        "ground_truth": "2",
        "response": "<think>Math..</think><description>Add 1 and 1</description><answer>2</answer>"
    }, {
        "prompt": "Capital of France?",
        "ground_truth": "Paris",
        "response": "<think>Geo..</think><description>Look up map</description><answer>London</answer>"
    }]
    
    try:
        results = compute_score(test_data)
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Test failed: {e}")