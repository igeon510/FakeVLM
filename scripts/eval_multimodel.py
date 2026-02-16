#!/usr/bin/env python3
import argparse
import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from PIL import Image
from tqdm import tqdm


REAL_LABEL = 1
FAKE_LABEL = 0


@dataclass
class Sample:
    sample_id: str
    image_path: Path
    label: int
    category: str
    question: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate local VLM checkpoints and OpenAI model on one test set."
    )
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--our-model-path", type=str, default=None)
    parser.add_argument("--fakevlm-model-path", type=str, default=None)
    parser.add_argument("--processor-path", type=str, default="llava-hf/llava-1.5-7b-hf")

    parser.add_argument("--openai-model", type=str, default=None)
    parser.add_argument("--openai-api-key", type=str, default=None)
    parser.add_argument("--openai-base-url", type=str, default="https://api.openai.com/v1")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_samples(test_json: Path, image_root: Path) -> List[Sample]:
    with test_json.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("test-json must be a JSON array")

    samples: List[Sample] = []
    for i, row in enumerate(rows):
        image_raw = row.get("image")
        if image_raw is None:
            raise KeyError(f"Missing image at index {i}")

        image_path = Path(image_raw)
        if not image_path.is_absolute():
            image_path = image_root / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        label = row.get("label")
        if label not in (0, 1):
            raise ValueError(f"Label must be 0/1 at index {i}, got: {label}")

        question = row.get("question")
        if question is None:
            convs = row.get("conversations", [])
            if convs and isinstance(convs, list) and isinstance(convs[0], dict):
                question = convs[0].get("value", "<image> Is this image real or fake?")
            else:
                question = "<image> Is this image real or fake?"

        sample_id = str(row.get("id", i))
        category = str(row.get("cate", "youtube"))
        samples.append(
            Sample(
                sample_id=sample_id,
                image_path=image_path,
                label=label,
                category=category,
                question=str(question),
            )
        )
    return samples


def extract_label(text: str) -> Optional[int]:
    lower = text.lower()
    fake_pos = re.search(r"\bfake\b", lower)
    real_pos = re.search(r"\breal\b", lower)

    if fake_pos and not real_pos:
        return FAKE_LABEL
    if real_pos and not fake_pos:
        return REAL_LABEL
    if fake_pos and real_pos:
        return FAKE_LABEL if fake_pos.start() < real_pos.start() else REAL_LABEL
    return None


def f1_binary(labels: List[int], preds: List[int]) -> float:
    tp = fp = fn = 0
    for y, p in zip(labels, preds):
        if y == REAL_LABEL and p == REAL_LABEL:
            tp += 1
        elif y == FAKE_LABEL and p == REAL_LABEL:
            fp += 1
        elif y == REAL_LABEL and p == FAKE_LABEL:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_metrics(labels: List[int], preds: List[int], unknown: int) -> Dict[str, Any]:
    total = len(labels)
    correct = sum(int(y == p) for y, p in zip(labels, preds))
    acc = correct / total if total else 0.0

    real_total = sum(int(y == REAL_LABEL) for y in labels)
    fake_total = sum(int(y == FAKE_LABEL) for y in labels)
    real_correct = sum(int(y == REAL_LABEL and p == REAL_LABEL) for y, p in zip(labels, preds))
    fake_correct = sum(int(y == FAKE_LABEL and p == FAKE_LABEL) for y, p in zip(labels, preds))

    tp = sum(int(y == REAL_LABEL and p == REAL_LABEL) for y, p in zip(labels, preds))
    tn = sum(int(y == FAKE_LABEL and p == FAKE_LABEL) for y, p in zip(labels, preds))
    fp = sum(int(y == FAKE_LABEL and p == REAL_LABEL) for y, p in zip(labels, preds))
    fn = sum(int(y == REAL_LABEL and p == FAKE_LABEL) for y, p in zip(labels, preds))

    return {
        "num_samples": total,
        "accuracy": round(acc, 4),
        "f1_real_positive": round(f1_binary(labels, preds), 4),
        "unknown_prediction_count": unknown,
        "real_accuracy": round(real_correct / real_total, 4) if real_total else 0.0,
        "fake_accuracy": round(fake_correct / fake_total, 4) if fake_total else 0.0,
        "confusion_matrix": {
            "tp_real": tp,
            "tn_fake": tn,
            "fp_fake_as_real": fp,
            "fn_real_as_fake": fn,
        },
    }


def infer_with_local_model(
    model_path: str,
    processor_path: str,
    samples: List[Sample],
    device: str,
    max_new_tokens: int,
    temperature: float,
) -> List[Dict[str, Any]]:
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and device.startswith("cuda") else torch.float32
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(processor_path)
    model = model.to(device)
    model.eval()

    results: List[Dict[str, Any]] = []
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": max(temperature, 1e-5),
    }
    if temperature == 0:
        gen_kwargs.pop("temperature")

    for sample in tqdm(samples, desc=f"local:{model_path}"):
        image = Image.open(sample.image_path).convert("RGB")
        inputs = processor(
            text=sample.question,
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
        gen_ids = output_ids[0][prompt_len:]
        output_text = processor.decode(gen_ids, skip_special_tokens=True).strip()
        if not output_text:
            output_text = processor.decode(output_ids[0], skip_special_tokens=True).strip()
        pred = extract_label(output_text)
        results.append(
            {
                "id": sample.sample_id,
                "label": sample.label,
                "pred": pred if pred is not None else -1,
                "output": output_text,
                "image": sample.image_path.as_posix(),
                "category": sample.category,
            }
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def image_to_data_url(path: Path) -> str:
    mime = "image/jpeg"
    suffix = path.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def parse_openai_text(resp_json: Dict[str, Any]) -> str:
    output = resp_json.get("output", [])
    if isinstance(output, list):
        texts: List[str] = []
        for item in output:
            content = item.get("content", []) if isinstance(item, dict) else []
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") in {"output_text", "text"} and c.get("text"):
                        texts.append(str(c["text"]))
        if texts:
            return "\n".join(texts)
    if isinstance(resp_json.get("output_text"), str):
        return str(resp_json["output_text"])
    return json.dumps(resp_json, ensure_ascii=False)


def infer_with_openai(
    model_name: str,
    api_key: str,
    base_url: str,
    samples: List[Sample],
    max_new_tokens: int,
    temperature: float,
) -> List[Dict[str, Any]]:
    endpoint = base_url.rstrip("/") + "/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    results: List[Dict[str, Any]] = []
    for sample in tqdm(samples, desc=f"openai:{model_name}"):
        payload = {
            "model": model_name,
            "temperature": temperature,
            "max_output_tokens": max_new_tokens,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": sample.question},
                        {"type": "input_image", "image_url": image_to_data_url(sample.image_path)},
                    ],
                }
            ],
        }
        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        raw = response.json()
        output_text = parse_openai_text(raw)
        pred = extract_label(output_text)
        results.append(
            {
                "id": sample.sample_id,
                "label": sample.label,
                "pred": pred if pred is not None else -1,
                "output": output_text,
                "image": sample.image_path.as_posix(),
                "category": sample.category,
            }
        )
    return results


def summarize_prediction_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels: List[int] = []
    preds: List[int] = []
    unknown = 0
    for row in rows:
        labels.append(int(row["label"]))
        pred = int(row["pred"])
        if pred not in (0, 1):
            unknown += 1
            pred = FAKE_LABEL
        preds.append(pred)
    return compute_metrics(labels, preds, unknown)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        args.device = "cpu"

    samples = load_samples(args.test_json, args.image_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    evaluations: List[Tuple[str, List[Dict[str, Any]]]] = []
    if args.our_model_path:
        rows = infer_with_local_model(
            model_path=args.our_model_path,
            processor_path=args.processor_path,
            samples=samples,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        evaluations.append(("our_model", rows))

    if args.fakevlm_model_path:
        rows = infer_with_local_model(
            model_path=args.fakevlm_model_path,
            processor_path=args.processor_path,
            samples=samples,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        evaluations.append(("fakevlm", rows))

    if args.openai_model:
        api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("Set --openai-api-key to evaluate OpenAI model.")
        rows = infer_with_openai(
            model_name=args.openai_model,
            api_key=api_key,
            base_url=args.openai_base_url,
            samples=samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        evaluations.append((f"openai_{args.openai_model}", rows))

    if not evaluations:
        raise ValueError("No model selected. Set at least one of --our-model-path, --fakevlm-model-path, --openai-model")

    summary: Dict[str, Any] = {}
    for model_name, rows in evaluations:
        metrics = summarize_prediction_rows(rows)
        summary[model_name] = metrics
        save_json(args.output_dir / f"{model_name}_predictions.json", {"rows": rows, "metrics": metrics})
        print(f"[{model_name}] acc={metrics['accuracy']} f1={metrics['f1_real_positive']} unknown={metrics['unknown_prediction_count']}")

    save_json(args.output_dir / "summary.json", summary)
    print(f"Saved summary to: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
