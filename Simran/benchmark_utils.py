from typing import Dict, Any, List, Tuple
import time


def _safe_div(n, d):
    return n / d if d != 0 else 0


def _flatten_medications(data: Dict[str, Any]) -> List[Dict]:
    meds = data.get("medication", [])
    flat = []
    for med in meds:
        name = med.get("name", "").strip().lower()
        details = med.get("medicationDetails", [])
        for d in details:
            flat.append({
                "name": name,
                "dose": d.get("dose", ""),
                "dosage": d.get("dosage", ""),
                "freq": d.get("freq", ""),
                "dur": d.get("dur", "")
            })
    return flat


def extraction_coverage(pred: Dict, gt: Dict) -> float:
    gt_meds = _flatten_medications(gt)
    pred_meds = _flatten_medications(pred)

    gt_names = set([m["name"] for m in gt_meds if m["name"]])
    pred_names = set([m["name"] for m in pred_meds if m["name"]])

    correct = len(gt_names & pred_names)
    return _safe_div(correct, len(gt_names))


def structured_field_reliability(pred: Dict, gt: Dict) -> float:
    fields = ["name", "age", "gender", "date"]

    correct = 0
    total = 0

    for f in fields:
        if gt.get(f):
            total += 1
            if str(pred.get(f)).strip().lower() == str(gt.get(f)).strip().lower():
                correct += 1

    return _safe_div(correct, total)


def dosage_integrity(pred: Dict, gt: Dict) -> float:
    pred_meds = _flatten_medications(pred)
    gt_meds = _flatten_medications(gt)

    correct = 0
    total = 0

    for p, t in zip(pred_meds, gt_meds):
        if t["dose"] or t["dosage"]:
            total += 1

            if (
                str(p["dose"]).strip() == str(t["dose"]).strip()
                and str(p["dosage"]).strip().lower() == str(t["dosage"]).strip().lower()
            ):
                correct += 1

    return _safe_div(correct, total)


def spurious_detection(pred: Dict, gt: Dict) -> float:
    gt_meds = _flatten_medications(gt)
    pred_meds = _flatten_medications(pred)

    gt_names = set([m["name"] for m in gt_meds if m["name"]])
    pred_names = set([m["name"] for m in pred_meds if m["name"]])

    false_positives = pred_names - gt_names
    return _safe_div(len(false_positives), len(pred_names))


def semantic_error_rate(pred: Dict, gt: Dict) -> float:
    pred_meds = _flatten_medications(pred)
    gt_meds = _flatten_medications(gt)

    errors = 0
    total = 0

    for p, t in zip(pred_meds, gt_meds):
        if t["name"]:
            total += 1
            if p["name"] != t["name"]:
                errors += 1

    return _safe_div(errors, total)


def normalization_effectiveness(pred: Dict, gt: Dict) -> float:
    pred_meds = _flatten_medications(pred)
    gt_meds = _flatten_medications(gt)

    correct = 0
    total = 0

    for p, t in zip(pred_meds, gt_meds):
        if t["name"]:
            total += 1
            if p["name"].lower() == t["name"].lower():
                correct += 1

    return _safe_div(correct, total)


def semantic_structuring_score(pred: Dict, gt: Dict) -> float:
    score = 0
    total = 0

    for key in gt.keys():
        if gt[key]:
            total += 1
            if key in pred and pred[key]:
                score += 1

    return _safe_div(score, total)


def end_to_end_accuracy(metrics: Dict[str, float]) -> float:
    return sum(metrics.values()) / len(metrics)


def critical_risk(pred: Dict) -> int:
    meds = _flatten_medications(pred)

    for m in meds:
        if m["dose"] and not any(unit in m["dose"].lower() for unit in ["mg", "ml", "g"]):
            return 1

    return 0


def evaluate(pred: Dict, gt: Dict) -> Tuple[Dict[str, float], Dict[str, str]]:
    metrics = {}

    metrics["Extraction Coverage"] = extraction_coverage(pred, gt)
    metrics["Structured Field Reliability"] = structured_field_reliability(pred, gt)
    metrics["Dosage Integrity Score"] = dosage_integrity(pred, gt)
    metrics["Spurious Detection Ratio"] = spurious_detection(pred, gt)
    metrics["Semantic Error Rate"] = semantic_error_rate(pred, gt)
    metrics["Normalization Effectiveness"] = normalization_effectiveness(pred, gt)
    metrics["Semantic Structuring Score"] = semantic_structuring_score(pred, gt)

    metrics["End-to-End Accuracy"] = end_to_end_accuracy({
        k: v for k, v in metrics.items()
        if k != "Spurious Detection Ratio" and k != "Semantic Error Rate"
    })

    metrics["Critical Risk Incidents"] = critical_risk(pred)

    targets = {
        "Extraction Coverage": 0.85,
        "Structured Field Reliability": 0.85,
        "Dosage Integrity Score": 0.90,
        "Spurious Detection Ratio": 0.05,
        "Semantic Error Rate": 0.05,
        "Normalization Effectiveness": 0.90,
        "Semantic Structuring Score": 0.88,
        "End-to-End Accuracy": 0.85,
        "Critical Risk Incidents": 0
    }

    status = {}

    for k, v in metrics.items():
        if k == "Spurious Detection Ratio" or k == "Semantic Error Rate":
            status[k] = "PASS" if v <= targets[k] else "FAIL"
        elif k == "Critical Risk Incidents":
            status[k] = "PASS" if v == 0 else "FAIL"
        else:
            status[k] = "PASS" if v >= targets[k] else "FAIL"

    return metrics, status
