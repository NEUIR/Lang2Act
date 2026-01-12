import argparse
import ast
import json
import os
import logging
import re
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, Optional

from tqdm import tqdm
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

LLM_JUDGE_SYSTEM_PROMPT = """You are an expert evaluation system for a question answering chatbot.
You will be given a list of evaluation items. For each item, you will see a query, a reference answer, and a generated answer.
Your task is to evaluate the correctness of the generated answer for EACH item in the list.
Your response MUST be a sequence of judgments, one for each item in the list, each on a new line, and each formatted as <judge>True or False</judge>.
For example, if you are given 3 items, your response should look exactly like this:
<judge>True</judge>
<judge>False</judge>
<judge>True</judge>
Do not add any other text, explanations, or item numbers.
"""

write_lock = threading.Lock()

def as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, list) else [v]
        except Exception:
            return [s]
    return [x]

def get_meta(item: Dict[str, Any]) -> Dict[str, Any]:
    m = item.get("meta_info")
    return m if isinstance(m, dict) else {}

def _normalize_source_token(s: str) -> Optional[str]:
    """Map evidence_sources to the five canonical buckets; ignore others."""
    if s is None:
        return None
    ss = str(s).strip()
    if not ss:
        return None
    low = ss.lower()

    if low in {"table"}:
        return "Table"
    if low in {"chart", "plot"}:
        return "Chart"
    if low in {"figure", "photo", "image", "figure/photo/image"}:
        return "Figure"
    if low in {"generalized-text (layout)", "layout"}:
        return "Layout"
    if low in {"pure-text (plain-text)", "plain-text", "text", "pure-text"}:
        return "Text"

    return None

def extract_sources(item: Dict[str, Any]) -> List[str]:
    meta = get_meta(item)

    src = item.get("evidence_sources", None)
    if src is None:
        src = meta.get("evidence_sources", None)

    sources = as_list(src)
    norm = []
    for s in sources:
        k = _normalize_source_token(s)
        if k is not None:
            norm.append(k)

    out, seen = [], set()
    for s in norm:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out

def extract_query_type(item: Dict[str, Any]) -> Optional[str]:
    meta = get_meta(item)
    qt = item.get("query_type", None)
    if qt is None:
        qt = meta.get("query_type", None)
    if qt is None:
        return None

    s = str(qt).strip().lower()
    if not s:
        return None

    if "single" in s:
        return "single-hop"
    if "multi" in s:
        return "multi-hop"
    return None

def extract_gt_image_paths(item: Dict[str, Any]) -> List[str]:
    gt = item.get("gt_image_paths", None)
    if gt is None:
        gt = get_meta(item).get("gt_image_paths", None)
    gt_list = as_list(gt)
    return [str(p) for p in gt_list if p is not None]

def hop_type_by_gt_count(item: Dict[str, Any]) -> Optional[str]:
    n = len(extract_gt_image_paths(item))
    if n > 1:
        return "multi-hop"
    if n == 1:
        return "single-hop"
    return None

def calc_acc(items: List[Dict[str, Any]]) -> Tuple[float, int, int]:
    evaluated = [x for x in items if x.get("llm_judge_score") is not None]
    if not evaluated:
        return 0.0, 0, 0
    correct = int(sum(1 for x in evaluated if float(x["llm_judge_score"]) == 1.0))
    total = len(evaluated)
    return correct / total, correct, total

def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}"

def get_llm_scores_for_batch(batch_data, client, model_name, max_retries=3):
    if not batch_data:
        return []

    num_prompts = len(batch_data)
    user_message_parts = ["Please evaluate the following items and provide your judgment for each one on a new line."]

    for i, item in enumerate(batch_data):
        ref = item.get("gt_answer") or item.get("answer") or ""
        gen = item.get("predicted_answer") or item.get("generated_answer") or ""
        q = item.get("question") or item.get("query") or ""

        item_text = (
            f"\n--- ITEM {i+1} ---\n"
            f"## Query\n{q}\n\n"
            f"## Reference Answer\n{ref}\n\n"
            f"## Generated Answer\n{gen}"
        )
        user_message_parts.append(item_text)

    user_message = "".join(user_message_parts)

    scores = [0.0] * num_prompts
    success = False

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=max(512, num_prompts * 20),
                temperature=0.0
            )
            full_response_text = response.choices[0].message.content

            judges = re.findall(r'<judge>(True|False|true|false)</judge>', full_response_text)

            if len(judges) == num_prompts:
                scores = [1.0 if j.lower() == "true" else 0.0 for j in judges]
                success = True
                break
            else:
                logging.warning(
                    f"Count mismatch (attempt {attempt+1}/{max_retries}): "
                    f"expected {num_prompts}, got {len(judges)}. Retrying..."
                )
        except Exception as e:
            logging.warning(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    if not success:
        logging.error("Batch scoring failed; defaulting scores to 0.")

    scored_batch = []
    for item, score in zip(batch_data, scores):
        new_item = item.copy()
        new_item["llm_judge_score"] = score
        scored_batch.append(new_item)

    return scored_batch

def derive_output_file(input_file: str) -> str:
    directory = os.path.dirname(os.path.abspath(input_file))
    base = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(directory, f"{base}_scored.jsonl")

def analyze_and_report(scored_file: str):
    if not os.path.exists(scored_file):
        logging.error(f"Scored file not found: {scored_file}")
        return

    all_items = []
    with open(scored_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                all_items.append(json.loads(line))
            except Exception:
                continue

    if not all_items:
        print("File is empty or could not be parsed.")
        return

    overall_acc, overall_c, overall_t = calc_acc(all_items)

    print("\n" + "=" * 80)
    print(f"Analysis file: {os.path.basename(scored_file)}")
    print("=" * 80)
    print(f"Overall Accuracy: {fmt_pct(overall_acc)}% ({overall_c}/{overall_t})")

    has_source = any(len(extract_sources(x)) > 0 for x in all_items)

    if has_source:
        source_buckets = defaultdict(list)
        for item in all_items:
            sources = extract_sources(item)
            for s in set(sources):
                source_buckets[s].append(item)

        text_acc, _, _ = calc_acc(source_buckets.get("Text", []))
        table_acc, _, _ = calc_acc(source_buckets.get("Table", []))
        chart_acc, _, _ = calc_acc(source_buckets.get("Chart", []))
        figure_acc, _, _ = calc_acc(source_buckets.get("Figure", []))
        layout_acc, _, _ = calc_acc(source_buckets.get("Layout", []))

        print("\n[Accuracy by evidence_sources (%)]")
        print("  +" + "-"*66 + "+")
        print(f"  | {'Text':<10} | {'Table':<10} | {'Chart':<10} | {'Figure':<10} | {'Layout':<10} |")
        print("  +" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+")
        print(
            f"  | {fmt_pct(text_acc):<10} | "
            f"{fmt_pct(table_acc):<10} | "
            f"{fmt_pct(chart_acc):<10} | "
            f"{fmt_pct(figure_acc):<10} | "
            f"{fmt_pct(layout_acc):<10} |"
        )
        print("  +" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+")
        return

    has_query_type = any(extract_query_type(x) is not None for x in all_items)
    if has_query_type:
        qt_buckets = {"single-hop": [], "multi-hop": []}
        for item in all_items:
            qt = extract_query_type(item)
            if qt in qt_buckets:
                qt_buckets[qt].append(item)

        sin_acc, sin_c, sin_t = calc_acc(qt_buckets["single-hop"])
        mul_acc, mul_c, mul_t = calc_acc(qt_buckets["multi-hop"])

        print("\n[Accuracy by query_type (%)]")
        print(f"  - single-hop : {fmt_pct(sin_acc)}% ({sin_c}/{sin_t})")
        print(f"  - multi-hop  : {fmt_pct(mul_acc)}% ({mul_c}/{mul_t})")
        return

    buckets = {"single-hop": [], "multi-hop": []}
    for item in all_items:
        ht = hop_type_by_gt_count(item)
        if ht in buckets:
            buckets[ht].append(item)

    sin_acc, sin_c, sin_t = calc_acc(buckets["single-hop"])
    mul_acc, mul_c, mul_t = calc_acc(buckets["multi-hop"])

    print("\n[Accuracy by GT image count (%)]")
    print(f"  - single-hop : {fmt_pct(sin_acc)}% ({sin_c}/{sin_t})")
    print(f"  - multi-hop  : {fmt_pct(mul_acc)}% ({mul_c}/{mul_t})")

def run_evaluation(args):
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        logging.error("Missing required environment variable SILICONFLOW_API_KEY.")
        return

    try:
        client = OpenAI(api_key=api_key, base_url=args.api_base_url)
        logging.info(f"API client initialized (workers={args.num_workers}).")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        return

    output_file = derive_output_file(args.input_file)

    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            all_data = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        logging.error(f"Failed to read input file: {e}")
        return

    processed = set()
    if args.resume and os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        processed.add(json.loads(line).get("uid"))
                    except Exception:
                        continue
            logging.info(f"Resume mode: skipping {len(processed)} processed items.")
        except Exception as e:
            logging.error(f"Failed to read resume file: {e}")

    items_to_process = [x for x in all_data if x.get("uid") not in processed]

    if not items_to_process:
        logging.info("All items already scored; running analysis.")
        analyze_and_report(output_file)
        return

    logging.info(
        f"Starting concurrent scoring: pending {len(items_to_process)}, "
        f"batch={args.batch_size}, workers={args.num_workers}"
    )

    batches = [items_to_process[i:i + args.batch_size] for i in range(0, len(items_to_process), args.batch_size)]

    with open(output_file, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_batch = {
                executor.submit(get_llm_scores_for_batch, batch, client, args.llm_judge_model, args.max_retries): batch
                for batch in batches
            }

            for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="scoring"):
                scored_batch = future.result()
                with write_lock:
                    for item in scored_batch:
                        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f_out.flush()

    logging.info(f"Scoring complete. Results saved to: {output_file}")
    analyze_and_report(output_file)

def main():
    parser = argparse.ArgumentParser(
        description="VQA LLM evaluation: prefer evidence_sources, else query_type, else GT image count."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL produced by inference.")
    parser.add_argument("--api_base_url", type=str, default="https://api.siliconflow.cn/v1/", help="API base URL.")
    parser.add_argument("--llm_judge_model", type=str, default="Qwen/Qwen2.5-72B-Instruct", help="Judge model name.")
    parser.add_argument("--batch_size", type=int, default=5, help="Items per request.")
    parser.add_argument("--num_workers", type=int, default=2, help="Concurrent threads.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing scored file.")
    parser.add_argument("--max_retries", type=int, default=3, help="Retry count per batch.")
    args = parser.parse_args()

    run_evaluation(args)

if __name__ == "__main__":
    main()
