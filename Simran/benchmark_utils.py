from typing import Dict, Any, List, Tuple
from rapidfuzz import fuzz


def _safe_div(n, d):
    return n / d if d != 0 else 0.0


def _is_present(value):
    return value not in ["", [], {}, None]


def _normalize_text(value: str) -> str:
    return str(value).strip().lower()


def _medicine_name_match(a: str, b: str, threshold: int = 80) -> bool:
    a = _normalize_text(a)
    b = _normalize_text(b)

    if not a or not b:
        return False

    if a == b:
        return True

    return fuzz.ratio(a, b) >= threshold


def _flatten_medications(data: Dict[str, Any]) -> List[Dict[str, str]]:
    meds = data.get("medication", [])
    flat = []

    for med in meds:
        name = _normalize_text(med.get("name", ""))
        details = med.get("medicationDetails", [])

        if not details:
            flat.append(
                {
                    "name": name,
                    "dose": "",
                    "dosage": "",
                    "freq": "",
                    "dur": "",
                }
            )
            continue

        for d in details:
            flat.append(
                {
                    "name": name,
                    "dose": str(d.get("dose", "")).strip(),
                    "dosage": str(d.get("dosage", "")).strip(),
                    "freq": str(d.get("freq", "")).strip(),
                    "dur": str(d.get("dur", "")).strip(),
                }
            )

    return flat


def _count_fuzzy_matches(gt_names: List[str], pred_names: List[str], threshold: int = 80) -> int:
    matched_pred_indices = set()
    correct = 0

    for gt_name in gt_names:
        for idx, pred_name in enumerate(pred_names):
            if idx in matched_pred_indices:
                continue

            if _medicine_name_match(gt_name, pred_name, threshold=threshold):
                correct += 1
                matched_pred_indices.add(idx)
                break

    return correct


def extraction_coverage(pred: Dict[str, Any], gt: Dict[str, Any]) -> float:
    gt_names = [m["name"] for m in _flatten_medications(gt) if _is_present(m["name"])]
    pred_names = [m["name"] for m in _flatten_medications(pred) if _is_present(m["name"])]

    if len(gt_names) == 0:
        return 1.0

    correct = _count_fuzzy_matches(gt_names, pred_names)
    return _safe_div(correct, len(gt_names))


def structured_field_reliability(pred: Dict[str, Any], gt: Dict[str, Any]) -> float:
    fields = ["name", "age", "gender", "date", "doctorUsername"]

    correct = 0
    total = 0

    for f in fields:
        gt_val = gt.get(f)
        if _is_present(gt_val):
            total += 1
            pred_val = pred.get(f, "")
            if _normalize_text(pred_val) == _normalize_text(gt_val):
                correct += 1

    if total == 0:
        return 1.0

    return _safe_div(correct, total)


def dosage_integrity(pred: Dict[str, Any], gt: Dict[str, Any]) -> float:
    pred_meds = _flatten_medications(pred)
    gt_meds = _flatten_medications(gt)

    total = 0
    correct = 0

    for t in gt_meds:
        if _is_present(t["dose"]) or _is_present(t["dosage"]) or _is_present(t["freq"]):
            total += 1

            matched = False
            for p in pred_meds:
                if _medicine_name_match(p["name"], t["name"]):
                    dose_ok = (
                        not _is_present(t["dose"])
                        or _normalize_text(p["dose"]) == _normalize_text(t["dose"])
                    )

                    dosage_ok = (
                        not _is_present(t["dosage"])
                        or _normalize_text(p["dosage"]) == _normalize_text(t["dosage"])
                    )

                    freq_ok = (
                        not _is_present(t["freq"])
                        or _normalize_text(p["freq"]) == _normalize_text(t["freq"])
                    )

                    # if GT dosage exists, dosage must match
                    # if GT freq exists, freq should also match when present
                    if dose_ok and dosage_ok and freq_ok:
                        matched = True
                        break

            if matched:
                correct += 1

    if total == 0:
        return 1.0

    return _safe_div(correct, total)


def spurious_detection(pred: Dict[str, Any], gt: Dict[str, Any]) -> float:
    gt_names = [m["name"] for m in _flatten_medications(gt) if _is_present(m["name"])]
    pred_names = [m["name"] for m in _flatten_medications(pred) if _is_present(m["name"])]

    if len(pred_names) == 0:
        return 0.0

    false_positive_count = 0

    for pred_name in pred_names:
        if not any(_medicine_name_match(pred_name, gt_name) for gt_name in gt_names):
            false_positive_count += 1

    return _safe_div(false_positive_count, len(pred_names))


def semantic_error_rate(pred: Dict[str, Any], gt: Dict[str, Any]) -> float:
    gt_names = [m["name"] for m in _flatten_medications(gt) if _is_present(m["name"])]
    pred_names = [m["name"] for m in _flatten_medications(pred) if _is_present(m["name"])]

    if len(gt_names) == 0:
        return 0.0

    correct = _count_fuzzy_matches(gt_names, pred_names)
    errors = len(gt_names) - correct

    return _safe_div(errors, len(gt_names))


def normalization_effectiveness(pred: Dict[str, Any], gt: Dict[str, Any]) -> float:
    gt_names = [m["name"] for m in _flatten_medications(gt) if _is_present(m["name"])]
    pred_names = [m["name"] for m in _flatten_medications(pred) if _is_present(m["name"])]

    if len(gt_names) == 0:
        return 1.0

    correct = _count_fuzzy_matches(gt_names, pred_names)
    return _safe_div(correct, len(gt_names))


def semantic_structuring_score(pred: Dict[str, Any], gt: Dict[str, Any]) -> float:
    score = 0
    total = 0

    for key, gt_val in gt.items():
        if _is_present(gt_val):
            total += 1
            pred_val = pred.get(key)
            if _is_present(pred_val):
                score += 1

    if total == 0:
        return 1.0

    return _safe_div(score, total)


def end_to_end_accuracy(metrics: Dict[str, float]) -> float:
    valid_values = [v for v in metrics.values() if isinstance(v, (int, float))]
    if len(valid_values) == 0:
        return 1.0
    return _safe_div(sum(valid_values), len(valid_values))


def critical_risk(pred: Dict[str, Any]) -> int:
    meds = _flatten_medications(pred)

    for m in meds:
        dose = _normalize_text(m["dose"])
        if dose and not any(unit in dose for unit in ["mg", "ml", "g", "mcg"]):
            return 1

    return 0


def processing_latency_score(processing_time_sec: float) -> float:
    return 1.0 if processing_time_sec <= 30 else 0.0


def evaluate(
    pred: Dict[str, Any],
    gt: Dict[str, Any],
    processing_time_sec: float = 0.0,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    metrics: Dict[str, float] = {}

    metrics["Extraction Coverage"] = extraction_coverage(pred, gt)
    metrics["Structured Field Reliability"] = structured_field_reliability(pred, gt)
    metrics["Dosage Integrity Score"] = dosage_integrity(pred, gt)
    metrics["Spurious Detection Ratio"] = spurious_detection(pred, gt)
    metrics["Semantic Error Rate"] = semantic_error_rate(pred, gt)
    metrics["Normalization Effectiveness"] = normalization_effectiveness(pred, gt)
    metrics["Semantic Structuring Score"] = semantic_structuring_score(pred, gt)

    metrics["End-to-End Accuracy"] = end_to_end_accuracy(
        {
            "Extraction Coverage": metrics["Extraction Coverage"],
            "Structured Field Reliability": metrics["Structured Field Reliability"],
            "Dosage Integrity Score": metrics["Dosage Integrity Score"],
            "Normalization Effectiveness": metrics["Normalization Effectiveness"],
            "Semantic Structuring Score": metrics["Semantic Structuring Score"],
        }
    )

    metrics["Critical Risk Incidents"] = critical_risk(pred)
    metrics["Processing Latency"] = processing_latency_score(processing_time_sec)

    targets = {
        "Extraction Coverage": 0.85,
        "Structured Field Reliability": 0.85,
        "Dosage Integrity Score": 0.90,
        "Spurious Detection Ratio": 0.05,
        "Semantic Error Rate": 0.05,
        "Normalization Effectiveness": 0.90,
        "Semantic Structuring Score": 0.88,
        "End-to-End Accuracy": 0.85,
        "Critical Risk Incidents": 0,
        "Processing Latency": 1.0,
    }

    status: Dict[str, str] = {}

    for k, v in metrics.items():
        if k in ["Spurious Detection Ratio", "Semantic Error Rate"]:
            status[k] = "PASS" if v <= targets[k] else "FAIL"
        elif k == "Critical Risk Incidents":
            status[k] = "PASS" if v == 0 else "FAIL"
        elif k == "Processing Latency":
            status[k] = "PASS" if processing_time_sec <= 30 else "FAIL"
        else:
            status[k] = "PASS" if v >= targets[k] else "FAIL"

    return metrics, status
