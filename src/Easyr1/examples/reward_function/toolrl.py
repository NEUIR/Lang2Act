"""
GRPO Reward Function (Strict Version - SiliconFlow Adapted)
- Strict Validation: <think> -> <description> -> <answer> order and <tool> isolation
- Parallel Evaluation: 8 threads concurrently calling Qwen2.5-72B-Instruct
- Required Environment Variable: SILICONFLOW_API_KEY
"""

import re
import os
import json
import logging
import time
import concurrent.futures
from typing import Dict, List, Any, Optional

# ==============================================================================
# --- Dependency Auto-Check ---
# ==============================================================================
try:
    import openai
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    import openai

# ==============================================================================
# --- Global Config & Client Initialization ---
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("RewardFunc_Strict_Silicon")

# SiliconFlow Configuration
API_KEY = os.environ.get("SILICONFLOW_API_KEY")
API_BASE_URL = "https://api.siliconflow.cn/v1/"
LLM_JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct"  

# Concurrency Configuration
PARALLEL_WORKERS = 8 
LLM_JUDGE_BATCH_SIZE = 10 

if not API_KEY:
    log.error("CRITICAL: SILICONFLOW_API_KEY is missing!")
    raise ValueError("SILICONFLOW_API_KEY is required.")

try:
    # Initialize single global client
    client = openai.OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    log.info(f"✅ SiliconFlow Client Initialized. Model: {LLM_JUDGE_MODEL}")
except Exception as e:
    log.error(f"Failed to init client: {e}")
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

# ==============================================================================
# --- Parallel Processing Worker ---
# ==============================================================================
def evaluate_chunk_worker(chunk_data: Dict[str, Any]) -> List[float]:
    """Task executed by a single thread: processes a batch of data"""
    chunk = chunk_data["items"]
    chunk_id = chunk_data["chunk_id"]
    
    if not chunk: return []

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

    # Retry Logic
    MAX_RETRIES = 3
    RETRY_DELAY = 1

    for attempt in range(MAX_RETRIES):
        try:
            # Call SiliconFlow API
            response = client.chat.completions.create(
                model=LLM_JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=len(chunk) * 10,  # Estimate Token
                temperature=0.0,
                timeout=60
            )
            full_text = (response.choices[0].message.content or "").strip()
            judges = re.findall(r"<judge>(True|False|true|false)</judge>", full_text)
            
            if len(judges) == len(chunk):
                return [1.0 if j.lower() == "true" else 0.0 for j in judges]
            else:
                log.warning(f"[Chunk {chunk_id}] Count mismatch: Got {len(judges)}, expected {len(chunk)}. Text: {full_text[:50]}...")
        
        except Exception as e:
            log.warning(f"[Chunk {chunk_id}] Attempt {attempt+1} failed: {e}")
        
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY * (2**attempt))

    log.error(f"[Chunk {chunk_id}] All retries failed. Defaulting to 0.0")
    return [0.0] * len(chunk)

# ==============================================================================
# --- Parallel Scheduler ---
# ==============================================================================
def get_llm_answer_scores_parallel(prompts_for_eval: List[Dict]) -> List[float]:
    if not prompts_for_eval: return []

    # Split data chunks
    chunks = []
    total_items = len(prompts_for_eval)
    for i in range(0, total_items, LLM_JUDGE_BATCH_SIZE):
        chunks.append({
            "items": prompts_for_eval[i : i + LLM_JUDGE_BATCH_SIZE],
            "chunk_id": i // LLM_JUDGE_BATCH_SIZE
        })
    
    log.info(f"Parallel Scoring: {total_items} items -> {len(chunks)} chunks (Workers={PARALLEL_WORKERS})")
    all_scores = []
    
    # Thread pool concurrent execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        # map guarantees result order matches chunks order
        results = executor.map(evaluate_chunk_worker, chunks)
        for res in results:
            all_scores.extend(res)
            
    return all_scores

# ==============================================================================
# --- Strict Regex Logic ---
# ==============================================================================
TOOL_TAG_PATTERN = re.compile(r"<tool\s+name=[\"'](?P<name>[^\"']+)[\"']", re.IGNORECASE)
FULL_TOOL_REGEX = re.compile(
    r"<tool\s+name=[\"'](?P<name>[^\"']+)[\"']\s+args=[\"'](?P<args>[^\"']+)[\"']\s*>(?P<body>.*?)</tool>",
    re.DOTALL | re.IGNORECASE,
)
IMG_ID_REGEX = re.compile(r"Image\s+(\d+)", re.IGNORECASE)

ALLOWED_TOOLS = {
    "locate_visual_element", "read_text_element", "read_numeric_value",
    "identify_entity_attribute", "compare_values", "compute_difference",
    "compute_percentage", "infer_missing_information",
}

def check_strict_format(response: str) -> Dict[str, Any]:
    """Strictly check XML structure, order, and tool leakage"""
    errors = []
    tags = ["think", "description", "answer"]
    positions = {}
    
    # 1. Tag existence and uniqueness
    for tag in tags:
        start_matches = list(re.finditer(f"<{tag}>", response, re.IGNORECASE))
        end_matches = list(re.finditer(f"</{tag}>", response, re.IGNORECASE))
        
        if len(start_matches) != 1 or len(end_matches) != 1:
            return {"valid": False, "errors": [f"Tag <{tag}> must appear exactly once."], "parsed": {}}
        
        positions[f"{tag}_start"] = start_matches[0].end()
        positions[f"{tag}_end"] = end_matches[0].start()
        
        if start_matches[0].start() >= end_matches[0].start():
             return {"valid": False, "errors": [f"Tag <{tag}> malformed."], "parsed": {}}

    # 2. Order check
    if positions["think_end"] > positions["description_start"]:
        errors.append("Order Error: <description> must follow <think>.")
    if positions["description_end"] > positions["answer_start"]:
        errors.append("Order Error: <answer> must follow <description>.")

    think_content = response[positions["think_start"]:positions["think_end"]].strip()
    desc_content = response[positions["description_start"]:positions["description_end"]].strip()
    answer_content = response[positions["answer_start"]:positions["answer_end"]].strip()

    if errors:
        return {"valid": False, "errors": errors, "parsed": {}}

    # 3. Tool Leakage Check
    total_tool_count = len(TOOL_TAG_PATTERN.findall(response))
    desc_tool_count = len(TOOL_TAG_PATTERN.findall(desc_content))
    
    if total_tool_count != desc_tool_count:
        errors.append(f"Tool Leakage: {total_tool_count} total tools, but only {desc_tool_count} inside <description>.")
        return {"valid": False, "errors": errors, "parsed": {}}

    return {
        "valid": True, 
        "errors": [],
        "parsed": {"think": think_content, "description": desc_content, "answer": answer_content}
    }

def validate_tool_content(description: str, think_content: str, num_images: int) -> float:
    """Score tool content quality"""
    tools = []
    for m in FULL_TOOL_REGEX.finditer(description):
        tools.append({"name": m.group("name").strip(), "args": m.group("args").strip(), "body": m.group("body").strip()})
    
    if not tools: return 0.0

    score = 1.0
    selected_images = {int(x) for x in IMG_ID_REGEX.findall(think_content)}
    PURE_NUMERIC_TOOLS = {"compute_difference", "compute_percentage"}
    
    for tool in tools:
        name = tool.get("name", "")
        args = tool.get("args", "")
        body = tool.get("body", "")
        
        if name not in ALLOWED_TOOLS:
            score -= 0.3
            continue
            
        if name in PURE_NUMERIC_TOOLS:
            if not args: score -= 0.2
        else:
            if not re.match(r"Image\s+\d+\s*:", args): score -= 0.2
            ref_ids = {int(x) for x in IMG_ID_REGEX.findall(args)}
            if not ref_ids:
                score -= 0.2
            else:
                for img_id in ref_ids:
                    if not (1 <= img_id <= num_images): score -= 0.5 
                    elif img_id not in selected_images: score -= 0.1 

        if not body: score -= 0.1

    if len(tools) > 8: score -= 0.1
    return max(0.0, score)

# ==============================================================================
# --- Main Entry: Compute Score ---
# ==============================================================================
def compute_score(reward_inputs: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    GRPO Reward Function Main Entry
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Expects list of dicts.")

    num_items = len(reward_inputs)
    if not num_items: return []

    # Weight Configuration
    W_ANSWER = 0.8
    W_FORMAT = 0.1
    W_TOOL = 0.1
    
    format_results = [] 
    tool_scores = []    
    prompts_eval = []
    indices_needing_eval = []

    # 1. Local Fast Validation
    for i, item in enumerate(reward_inputs):
        response = item.get("response", "")
        
        # A. Format Validation (Hard Constraint)
        fmt_check = check_strict_format(response)
        
        if not fmt_check["valid"]:
            format_results.append(0.0)
            tool_scores.append(0.0)
            continue 
        
        # B. Format Passed
        format_results.append(1.0)
        parsed = fmt_check["parsed"]
        
        # C. Tool Scoring
        t_score = validate_tool_content(
            parsed["description"], 
            parsed["think"], 
            int(item.get("num_images", 3))
        )
        tool_scores.append(t_score)
        
        # D. Prepare LLM Evaluation
        if parsed["answer"]:
            prompts_eval.append({
                "query": item.get("prompt"), 
                "reference_answer": item.get("ground_truth"), 
                "generated_answer": parsed["answer"]
            })
            indices_needing_eval.append(i)

    # 2. Parallel LLM Evaluation
    answer_scores = [0.0] * num_items
    
    if prompts_eval:
        try:
            start_t = time.time()
            llm_scores = get_llm_answer_scores_parallel(prompts_eval)
            log.info(f"Parallel Eval: {len(prompts_eval)} items in {time.time() - start_t:.2f}s")
            
            for s, idx in zip(llm_scores, indices_needing_eval):
                answer_scores[idx] = s
        except Exception as e:
            log.error(f"LLM Judge failed: {e}")

    # 3. Result Aggregation
    final_scores = []
    for i in range(num_items):
        if format_results[i] == 0.0:
            overall = 0.0
        else:
            overall = (W_ANSWER * answer_scores[i]) + (W_FORMAT * format_results[i]) + (W_TOOL * tool_scores[i])
            overall = max(0.0, min(1.0, overall))
            
        final_scores.append({
            "overall": round(overall, 6),
            "answer_score": answer_scores[i],
            "format_score": format_results[i],
            "tool_score": tool_scores[i] if format_results[i] > 0 else 0.0
        })

    # Sample Log
    if final_scores:
        log.info(json.dumps({"sample_0": final_scores[0]}, ensure_ascii=False))

    return final_scores

# Test Entry
if __name__ == "__main__":
    pass