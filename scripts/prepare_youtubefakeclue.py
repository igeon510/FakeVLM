#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_QUESTION = "<image> Is this image real or fake?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YoutubeFakeClue-style metadata to FakeVLM train/eval JSON formats."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-key", type=str, default="image")
    parser.add_argument("--label-key", type=str, default="label")
    parser.add_argument("--explanation-key", type=str, default="explanation")
    parser.add_argument("--id-key", type=str, default="id")
    parser.add_argument("--question-key", type=str, default="question")
    parser.add_argument("--category-key", type=str, default="cate")
    parser.add_argument("--default-category", type=str, default="youtube")
    parser.add_argument("--default-question", type=str, default=DEFAULT_QUESTION)
    return parser.parse_args()


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def normalize_label(raw: Any) -> Tuple[int, str]:
    if isinstance(raw, bool):
        return (1 if raw else 0), ("real" if raw else "fake")
    if isinstance(raw, int):
        if raw not in (0, 1):
            raise ValueError(f"Unsupported integer label: {raw}")
        return raw, ("real" if raw == 1 else "fake")
    if isinstance(raw, str):
        t = raw.strip().lower()
        if t in {"real", "1", "true", "human", "authentic"}:
            return 1, "real"
        if t in {"fake", "0", "false", "ai", "deepfake", "synthetic"}:
            return 0, "fake"
    raise ValueError(f"Unsupported label value: {raw}")


def resolve_image_path(dataset_root: Path, image_value: str, label_text: str) -> Path:
    candidate = Path(image_value)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    direct = dataset_root / candidate
    if direct.exists():
        return direct
    by_label = dataset_root / label_text / candidate.name
    if by_label.exists():
        return by_label
    raise FileNotFoundError(
        f"Image not found for value '{image_value}'. Tried: {direct}, {by_label}"
    )


def make_answer(label_text: str, explanation: Optional[str]) -> str:
    base = f"This is a {label_text} image."
    if explanation:
        exp = explanation.strip()
        if exp:
            return f"{base} {exp}"
    return base


def to_relative_posix(base: Path, path: Path) -> str:
    try:
        rel = path.resolve().relative_to(base.resolve())
        return rel.as_posix()
    except Exception:
        return path.as_posix()


def convert_split(
    items: List[Dict[str, Any]],
    dataset_root: Path,
    default_question: str,
    image_key: str,
    label_key: str,
    explanation_key: str,
    id_key: str,
    question_key: str,
    category_key: str,
    default_category: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sft_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(items):
        if image_key not in item:
            raise KeyError(f"Missing image key '{image_key}' at index {idx}")
        if label_key not in item:
            raise KeyError(f"Missing label key '{label_key}' at index {idx}")

        label_id, label_text = normalize_label(item[label_key])
        image_path = resolve_image_path(dataset_root, str(item[image_key]), label_text)
        image_rel = to_relative_posix(dataset_root, image_path)

        question = str(item.get(question_key, default_question))
        explanation = item.get(explanation_key)
        sample_id = item.get(id_key, str(idx))
        category = str(item.get(category_key, default_category))
        answer = make_answer(label_text, None if explanation is None else str(explanation))

        sft_rows.append(
            {
                "id": sample_id,
                "image": image_rel,
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer},
                ],
                "label": label_id,
                "cate": category,
            }
        )

        eval_rows.append(
            {
                "id": sample_id,
                "image": image_rel,
                "label": label_id,
                "cate": category,
                "conversations": [{"from": "human", "value": question}],
            }
        )

    return sft_rows, eval_rows


def dump_json(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()

    train_items = load_json(args.train_json)
    test_items = load_json(args.test_json)

    train_sft, train_eval = convert_split(
        items=train_items,
        dataset_root=dataset_root,
        default_question=args.default_question,
        image_key=args.image_key,
        label_key=args.label_key,
        explanation_key=args.explanation_key,
        id_key=args.id_key,
        question_key=args.question_key,
        category_key=args.category_key,
        default_category=args.default_category,
    )
    test_sft, test_eval = convert_split(
        items=test_items,
        dataset_root=dataset_root,
        default_question=args.default_question,
        image_key=args.image_key,
        label_key=args.label_key,
        explanation_key=args.explanation_key,
        id_key=args.id_key,
        question_key=args.question_key,
        category_key=args.category_key,
        default_category=args.default_category,
    )

    dump_json(output_dir / "train_sft.json", train_sft)
    dump_json(output_dir / "test_sft.json", test_sft)
    dump_json(output_dir / "train_eval.json", train_eval)
    dump_json(output_dir / "test_eval.json", test_eval)

    print(f"Saved: {output_dir / 'train_sft.json'} ({len(train_sft)} rows)")
    print(f"Saved: {output_dir / 'test_sft.json'} ({len(test_sft)} rows)")
    print(f"Saved: {output_dir / 'train_eval.json'} ({len(train_eval)} rows)")
    print(f"Saved: {output_dir / 'test_eval.json'} ({len(test_eval)} rows)")


if __name__ == "__main__":
    main()
