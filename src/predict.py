#!/usr/bin/env python
# coding: utf-8
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import argparse
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# Try importing prompt templates
try:
    from prompts import PROMPT_TEMPLATES
except ImportError:
    logging.warning("prompts.py not found, using default logic.")
    PROMPT_TEMPLATES = {}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class DefaultConfig:
    # HF defaults
    hf_namespace: str = "xiongyq"
    hf_jsonl: str = "top3_test.jsonl"
    hf_parquet: str = "images.parquet"
    repo_prefix: str = "Lang2Act-Test"

    # Output default 
    output_root: str = "result/"

    # Sweep defaults
    k_values: Tuple[int, ...] = (3,)
    prompt_modes: Tuple[str, ...] = ("Lang2Act",)

    # vLLM defaults
    batch_size: int = 20
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    limit_mm_per_prompt: int = 5
    gpu_memory_utilization: float = 0.90
    temperature: float = 0.0
    repetition_penalty: float = 1.05
    max_tokens: int = 2048
    dtype: str = "bfloat16"

    # Pixel limits
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 512 * 28 * 28

    # Save flags
    save_raw_default: bool = True
    save_used_images_default: bool = False


CFG = DefaultConfig()


# ==============================================================================
# Helpers
# ==============================================================================

def _parse_tag(text: str, tag: str) -> Optional[str]:
    """Extract content within XML tags using regex, case-insensitive and dotall."""
    if not isinstance(text, str):
        return None
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSONL file not found: {path}")
        
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            if s in {"{", "}", "[", "]"}:
                continue
            try:
                items.append(json.loads(s))
            except json.JSONDecodeError:
                logging.warning(f"Skipping malformed line {path}:{lineno}")
    logging.info(f"Loaded {len(items)} items from {path}.")
    return items


def get_uid(item: Dict[str, Any]) -> Optional[str]:
    for key in ["uid", "qa_id", "id"]:
        val = item.get(key)
        if val is not None and str(val).strip() != "":
            return str(val)
    return None


def get_question(item: Dict[str, Any]) -> Optional[str]:
    return item.get("question") or item.get("query")


def build_user_message(
    question: str,
    image_sources: List[Union[str, Image.Image]],
    prompt_mode: str
) -> List[Dict[str, Any]]:
    """Build prompt based on definitions in prompts.py"""
    num_images = len(image_sources)

    if prompt_mode not in PROMPT_TEMPLATES:
        logging.warning(
            f"Mode {prompt_mode} not defined in prompts.py, falling back to default format."
        )
        sys_text = f"Answer the question based on the {num_images} images."
        first_text = f"{sys_text}\n\nQuestion: {question}"
    else:
        tmpl = PROMPT_TEMPLATES[prompt_mode]

        if "{num_images}" in tmpl:
            sys_text = tmpl.format(num_images=num_images)
            first_text = f"{sys_text}\n\nQuestion: {question}".strip()
        else:
            sys_text = tmpl
            first_text = f"{sys_text}\n\nQuestion: {question}".strip()

    content: List[Dict[str, Any]] = [{"type": "text", "text": first_text}]
    for src in image_sources:
        content.append({"type": "image", "image": src})

    return content



def _resize_pil_by_pixels(img: Image.Image, min_pixels: int, max_pixels: int) -> Image.Image:
    if not isinstance(img, Image.Image):
        return img
    w, h = img.size
    s = max(1, w * h)

    if s > max_pixels:
        r = (max_pixels / s) ** 0.5
        w2, h2 = max(1, int(w * r)), max(1, int(h * r))
        img = img.resize((w2, h2), Image.Resampling.LANCZOS)
    elif s < min_pixels:
        r = (min_pixels / s) ** 0.5
        w2, h2 = max(1, int(w * r)), max(1, int(h * r))
        img = img.resize((w2, h2), Image.Resampling.LANCZOS)

    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _apply_pixel_limits_in_message(message: Dict[str, Any], min_pixels: int, max_pixels: int) -> Dict[str, Any]:
    new_content = []
    for part in message.get("content", []):
        if isinstance(part, dict) and part.get("type") == "image":
            img_src = part.get("image")
            try:
                pil = Image.open(img_src) if isinstance(img_src, str) else img_src
                pil = _resize_pil_by_pixels(pil, min_pixels, max_pixels)
                new_content.append({"type": "image", "image": pil})
            except Exception as e:
                logging.warning(f"Image processing failed: {e}; skipping.")
                new_content.append(part)
        else:
            new_content.append(part)
    return {"role": message.get("role", "user"), "content": new_content}


def extract_bucket_field(item: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    meta = item.get("meta_info") or {}
    res = {}
    
    if dataset == "mmlb":
        ev = item.get("evidence_sources") or meta.get("evidence_sources")
        if ev: res["evidence_sources"] = ev
    elif dataset == "vidoseek":
        qt = item.get("query_type") or meta.get("query_type")
        if qt: res["query_type"] = qt
    elif dataset == "slidevqa":
        gt = item.get("gt_image_paths")
        if gt: res["gt_image_paths"] = gt
        
    return res


def parse_predicted_answer(raw_text: str) -> str:
    """Only extract content within <answer> tag."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()
    if cleaned.lower().startswith("xml"):
        cleaned = cleaned[3:].strip()

    val = _parse_tag(cleaned, "answer")
    if val:
        return val
    return cleaned


def make_output_record(
    original_item: Dict[str, Any],
    dataset: str,
    model: str,
    top_k: int,
    prompt_mode: str,
    predicted_answer: str,
    used_topk_images: Optional[List[str]] = None,
    model_response_raw: Optional[str] = None,
) -> Dict[str, Any]:
    uid = get_uid(original_item) or ""
    question = get_question(original_item) or ""
    gt_answer = original_item.get("answer")
    bucket_fields = extract_bucket_field(original_item, dataset)

    record: Dict[str, Any] = {
        "uid": uid,
        "question": question,
        **bucket_fields,
        "gt_answer": gt_answer,
        "predicted_answer": predicted_answer,
        "dataset": dataset,
        "model": model,
        "top_k": top_k,
        "prompt_mode": prompt_mode,
    }

    if used_topk_images is not None:
        record["used_topk_images"] = used_topk_images
    if model_response_raw is not None:
        record["model_response_raw"] = model_response_raw

    return record


# ==============================================================================
# HF Parquet store
# ==============================================================================

def hf_download(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)


class ParquetImageStore:
    def __init__(self, parquet_path: str):
        if not os.path.exists(parquet_path):
             raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        from datasets import Dataset
        logging.info(f"Loading Parquet Index: {parquet_path}")
        self.ds = Dataset.from_parquet(parquet_path)

        self.key2idx: Dict[Tuple[str, str], int] = {}
        uids = self.ds["uid"]
        rels = self.ds["relpath"]
        
        for i in tqdm(range(len(self.ds)), desc="Building Index", unit="row"):
            self.key2idx[(str(uids[i]), rels[i])] = i

    def get_image(self, uid: str, relpath: str) -> Optional[Image.Image]:
        idx = self.key2idx.get((str(uid), relpath))
        if idx is None:
            return None
        try:
            return self.ds[int(idx)]["image"]
        except Exception:
            return None


def load_hf_dataset_and_store(hf_repo: str, hf_jsonl: str, hf_parquet: str) -> Tuple[List[Dict[str, Any]], ParquetImageStore]:
    local_jsonl_path = hf_download(hf_repo, hf_jsonl)
    local_parquet_path = hf_download(hf_repo, hf_parquet)
    data = load_jsonl(local_jsonl_path)
    store = ParquetImageStore(local_parquet_path)
    return data, store


# ==============================================================================
# Inference core
# ==============================================================================

def run_inference_once(
    *,
    data: List[Dict[str, Any]],
    store: ParquetImageStore,
    output_path: str,
    model_path: str,
    dataset: str,
    prompt_mode: str,
    top_k: int,
    batch_size: int,
    tensor_parallel_size: int,
    dtype: str,
    limit_mm_per_prompt: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    temperature: float,
    repetition_penalty: float,
    max_tokens: int,
    min_pixels: int,
    max_pixels: int,
    save_raw: bool,
    save_used_images: bool,
):
    logging.info(f"[RUN] dataset={dataset} prompt={prompt_mode} top_k={top_k} output={output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize vLLM
    logging.info(f"Loading Model: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    
    llm = LLM(
        trust_remote_code=True,
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        limit_mm_per_prompt={"image": limit_mm_per_prompt, "video": 0},
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
    )

    # Resume progress
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        done_uid = get_uid(json.loads(line))
                        if done_uid:
                            done.add(done_uid)
                    except Exception:
                        pass
    
    items_to_process = [it for it in data if get_uid(it) and get_uid(it) not in done]
    logging.info(f"Total {len(data)}, Done {len(done)}, Pending {len(items_to_process)}")

    with open(output_path, "a", encoding="utf-8") as fout, tqdm(
        total=len(items_to_process), desc=f"Inference {prompt_mode}", dynamic_ncols=True
    ) as pbar:

        for i in range(0, len(items_to_process), batch_size):
            batch = items_to_process[i: i + batch_size]

            msgs: List[List[Dict[str, Any]]] = []
            metas: List[Tuple[Dict[str, Any], List[str]]] = []

            for it in batch:
                uid = get_uid(it)
                question = get_question(it)
                
                top_paths = it.get("top3_image_paths") or it.get("retrieved_image_paths") or []
                top_paths = top_paths[: top_k]

                pil_imgs: List[Image.Image] = []
                used_relpaths: List[str] = []
                for rel in top_paths:
                    pil = store.get_image(uid=str(uid), relpath=rel)
                    if pil:
                        pil_imgs.append(pil)
                        used_relpaths.append(rel)
                
                if not pil_imgs:
                    logging.warning(f"Skipping uid={uid}: No valid images")
                    continue

                content = build_user_message(question, pil_imgs, prompt_mode)
                raw_user_msg = {"role": "user", "content": content}
                
                user_msg_with_pil = _apply_pixel_limits_in_message(raw_user_msg, min_pixels, max_pixels)

                msgs.append([user_msg_with_pil])
                metas.append((it, used_relpaths))

            if not msgs:
                pbar.update(len(batch))
                continue

            batch_inputs = []
            for msg in msgs:
                prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(msg)
                batch_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image_inputs}})

            outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

            for (original_item, used_relpaths), out in zip(metas, outputs):
                raw_text = out.outputs[0].text if out.outputs else ""
                
                predicted_answer = parse_predicted_answer(raw_text)

                record = make_output_record(
                    original_item=original_item,
                    dataset=dataset,
                    model=model_path,
                    top_k=top_k,
                    prompt_mode=prompt_mode,
                    predicted_answer=predicted_answer,
                    used_topk_images=(used_relpaths if save_used_images else None),
                    model_response_raw=(raw_text if save_raw else None),
                )

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

            pbar.update(len(batch))

    logging.info(f"Finished: {output_path}")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="HF Parquet Inference (vLLM)")

    parser.add_argument("--dataset", type=str, required=True, choices=["mmlb", "vidoseek", "slidevqa"])
    parser.add_argument("--model", type=str, required=True)

    # Added arguments for local files
    parser.add_argument("--local_jsonl", type=str, default=None, help="Local path to .jsonl file")
    parser.add_argument("--local_parquet", type=str, default=None, help="Local path to .parquet file")

    parser.add_argument("--hf_namespace", type=str, default=CFG.hf_namespace)
    parser.add_argument("--repo_prefix", type=str, default=CFG.repo_prefix)
    parser.add_argument("--hf_jsonl", type=str, default=CFG.hf_jsonl)
    parser.add_argument("--hf_parquet", type=str, default=CFG.hf_parquet)
    parser.add_argument("--output_root", type=str, default=CFG.output_root)

    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--prompt_mode", type=str, default=None) 
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--run_default_sweep", action="store_true")

    # vLLM params
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--tensor_parallel_size", type=int, default=CFG.tensor_parallel_size)
    parser.add_argument("--max_model_len", type=int, default=CFG.max_model_len)
    parser.add_argument("--limit_mm_per_prompt", type=int, default=CFG.limit_mm_per_prompt)
    parser.add_argument("--gpu_memory_utilization", type=float, default=CFG.gpu_memory_utilization)
    parser.add_argument("--temperature", type=float, default=CFG.temperature)
    parser.add_argument("--repetition_penalty", type=float, default=CFG.repetition_penalty)
    parser.add_argument("--max_tokens", type=int, default=CFG.max_tokens)
    parser.add_argument("--dtype", type=str, default=CFG.dtype)

    parser.add_argument("--min_pixels", type=int, default=CFG.min_pixels)
    parser.add_argument("--max_pixels", type=int, default=CFG.max_pixels)

    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument("--save_used_images", action="store_true")

    args = parser.parse_args()

    # 1. Load Data (Local or HF)
    if args.local_jsonl and args.local_parquet:
        logging.info(f"Using LOCAL files:\n  JSONL: {args.local_jsonl}\n  Parquet: {args.local_parquet}")
        data = load_jsonl(args.local_jsonl)
        store = ParquetImageStore(args.local_parquet)
    else:
        hf_repo = f"{args.hf_namespace}/{args.repo_prefix}-{args.dataset}"
        logging.info(f"Using Hugging Face repo: {hf_repo}")
        data, store = load_hf_dataset_and_store(hf_repo, args.hf_jsonl, args.hf_parquet)

    # 2. Prepare Configuration
    save_raw = True if args.save_raw else CFG.save_raw_default
    save_used_images = True if args.save_used_images else CFG.save_used_images_default
    
    common_kwargs = {
        "data": data,
        "store": store,
        "model_path": args.model,
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "limit_mm_per_prompt": args.limit_mm_per_prompt,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "max_tokens": args.max_tokens,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "save_raw": save_raw,
        "save_used_images": save_used_images,
    }

    # 3. Execution Mode (Sweep vs Single)
    if args.run_default_sweep:
        model_tag = os.path.basename(args.model.rstrip("/"))
        exp_name = f"{args.dataset}_{model_tag}"
        out_dir = os.path.join(args.output_root, exp_name)
        
        for k in CFG.k_values:
            for mode in CFG.prompt_modes:
                out_path = os.path.join(out_dir, f"predictions_{mode}_top{k}.jsonl")
                run_inference_once(
                    output_path=out_path,
                    prompt_mode=mode,
                    top_k=k,
                    **common_kwargs
                )
    else:
        # Single run
        prompt_mode = args.prompt_mode or (CFG.prompt_modes[0] if CFG.prompt_modes else "Lang2Act")
        top_k = args.top_k if args.top_k is not None else (CFG.k_values[0] if CFG.k_values else 3)
        
        if args.output:
            out_path = args.output
        else:
            model_tag = os.path.basename(args.model.rstrip("/"))
            exp_name = f"{args.dataset}_{model_tag}"
            out_dir = os.path.join(args.output_root, exp_name)
            out_path = os.path.join(out_dir, f"predictions_{prompt_mode}_top{top_k}.jsonl")

        run_inference_once(
            output_path=out_path,
            prompt_mode=prompt_mode,
            top_k=top_k,
            **common_kwargs
        )


if __name__ == "__main__":
    main()