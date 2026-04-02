import json
import re
from copy import deepcopy
from difflib import SequenceMatcher


# =========================================================
# Basic helpers
# =========================================================
def normalize_text(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def is_empty_value(value):
    if value is None:
        return True
    if isinstance(value, str) and normalize_text(value) == "":
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    if isinstance(value, dict):
        return all(is_empty_value(v) for v in value.values()) if value else True
    return False


def similarity(a, b):
    a = normalize_text(a)
    b = normalize_text(b)
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def normalize_date(value):
    text = normalize_text(value)
    if not text:
        return ""

    text = text.replace("/", "-").replace(".", "-").replace(",", " ")

    month_map = {
        "january": "01", "jan": "01",
        "february": "02", "feb": "02",
        "march": "03", "mar": "03",
        "april": "04", "apr": "04",
        "may": "05",
        "june": "06", "jun": "06",
        "july": "07", "jul": "07",
        "august": "08", "aug": "08",
        "september": "09", "sep": "09", "sept": "09",
        "october": "10", "oct": "10",
        "november": "11", "nov": "11",
        "december": "12", "dec": "12",
    }

    m = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{2,4})$", text)
    if m:
        d, mo, y = m.groups()
        return f"{d.zfill(2)}-{mo.zfill(2)}-{y}"

    tokens = text.split()
    if len(tokens) >= 3:
        month_num = month_map.get(tokens[0], "")
        if month_num and tokens[1].isdigit() and tokens[2].isdigit():
            return f"{tokens[1].zfill(2)}-{month_num}-{tokens[2]}"

    return text


def normalize_time(value):
    text = normalize_text(value)
    if not text:
        return ""
    text = text.replace(".", ":")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_numberish(value):
    text = normalize_text(value)
    if not text:
        return ""
    return re.sub(r"\s+", "", text)


def normalize_bp(value):
    text = normalize_text(value)
    if not text:
        return ""
    text = text.replace("mmhg", "").replace("mm of hg", "")
    text = re.sub(r"\s+", "", text)
    return text


def listify_text_block(value):
    if value is None:
        return []

    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = value.splitlines()
    else:
        items = [str(value)]

    cleaned = []
    for item in items:
        t = normalize_text(item)
        if t:
            cleaned.append(t)
    return cleaned


def normalize_field_value(field_name, value):
    field = normalize_text(field_name)

    if field in {"date", "followupdate"}:
        return normalize_date(value)

    if field in {"time"}:
        return normalize_time(value)

    if field in {"bp"}:
        return normalize_bp(value)

    if field in {
        "heartrate", "respiratoryrate", "respiratory_rate",
        "temp", "temperature", "spo2", "weight", "height",
        "bmi", "bsa", "headcircumference"
    }:
        return normalize_numberish(value)

    if field in {"diagnosis", "complaints", "notes", "test", "tests"}:
        return listify_text_block(value)

    return normalize_text(value)


# =========================================================
# Medication helpers
# =========================================================
def parse_medicine_line(line):
    result = {
        "name": "",
        "dose": "",
        "dosage": "",
        "route": "",
        "freq": "",
        "dur": "",
        "class": "",
        "when": ""
    }

    if not line:
        return result

    if isinstance(line, dict):
        result["name"] = normalize_text(line.get("name", ""))
        for key in ["dose", "dosage", "route", "freq", "dur", "class", "when"]:
            result[key] = normalize_text(line.get(key, ""))
        return result

    text = str(line)
    parts = [p.strip() for p in text.split("|")]

    if parts:
        result["name"] = normalize_text(parts[0])

    for part in parts[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            k = normalize_text(k)
            v = normalize_text(v)
            if k in result:
                result[k] = v

    return result


def normalize_medications(medications):
    normalized = []

    if medications is None:
        return normalized

    if isinstance(medications, str):
        lines = [x.strip() for x in medications.splitlines() if normalize_text(x)]
        for line in lines:
            normalized.append(parse_medicine_line(line))
        return normalized

    if isinstance(medications, list):
        for item in medications:
            if isinstance(item, dict) and "medicationDetails" in item:
                med_name = normalize_text(item.get("name", ""))
                details_list = item.get("medicationDetails", [])

                if not isinstance(details_list, list) or not details_list:
                    details_list = [{}]

                for detail in details_list:
                    if not isinstance(detail, dict):
                        detail = {}

                    normalized.append({
                        "name": med_name,
                        "dose": normalize_text(detail.get("dose", "")),
                        "dosage": normalize_text(detail.get("dosage", "")),
                        "route": normalize_text(detail.get("route", "")),
                        "freq": normalize_text(detail.get("freq", "")),
                        "dur": normalize_text(detail.get("dur", "")),
                        "class": normalize_text(detail.get("class", "")),
                        "when": normalize_text(detail.get("when", ""))
                    })

            elif isinstance(item, dict):
                normalized.append(parse_medicine_line(item))
            else:
                normalized.append(parse_medicine_line(item))

    return normalized


def medicine_name_match(pred_name, gt_name, threshold=0.80):
    pred_name = normalize_text(pred_name)
    gt_name = normalize_text(gt_name)

    if pred_name == gt_name:
        return True

    return similarity(pred_name, gt_name) >= threshold


# =========================================================
# Structured field evaluation
# =========================================================
def structured_field_reliability(pred, gt):
    skip_fields = {"medication", "medications", "vitals", "followup"}

    compared = 0
    correct = 0
    details = []

    for key, gt_value in gt.items():
        if key in skip_fields:
            continue

        if is_empty_value(gt_value):
            details.append({
                "field": key,
                "status": "skipped",
                "reason": "empty_in_ground_truth"
            })
            continue

        pred_value = pred.get(key, "")

        gt_norm = normalize_field_value(key, gt_value)
        pred_norm = normalize_field_value(key, pred_value)

        matched = gt_norm == pred_norm

        compared += 1
        if matched:
            correct += 1
            status = "correct"
        else:
            status = "wrong"

        details.append({
            "field": key,
            "status": status,
            "gt": gt_norm,
            "pred": pred_norm
        })

    score = correct / compared if compared > 0 else 0.0

    return {
        "score": score,
        "correct": correct,
        "total": compared,
        "details": details
    }


def compare_nested_dict(pred_block, gt_block, prefix):
    pred_block = pred_block or {}
    gt_block = gt_block or {}

    compared = 0
    correct = 0
    details = []

    gt_keys = set(gt_block.keys())

    for key in gt_keys:
        gt_value = gt_block.get(key, "")

        if is_empty_value(gt_value):
            details.append({
                "field": f"{prefix}.{key}",
                "status": "skipped",
                "reason": "empty_in_ground_truth"
            })
            continue

        pred_value = pred_block.get(key, "")

        gt_norm = normalize_field_value(key, gt_value)
        pred_norm = normalize_field_value(key, pred_value)

        matched = gt_norm == pred_norm

        compared += 1
        if matched:
            correct += 1
            status = "correct"
        else:
            status = "wrong"

        details.append({
            "field": f"{prefix}.{key}",
            "status": status,
            "gt": gt_norm,
            "pred": pred_norm
        })

    score = correct / compared if compared > 0 else 0.0

    return {
        "score": score,
        "correct": correct,
        "total": compared,
        "details": details
    }


# =========================================================
# Medicine metrics
# =========================================================
def extraction_coverage(pred_meds, gt_meds):
    if len(gt_meds) == 0:
        return {
            "score": 0.0,
            "matched": 0,
            "total_gt": 0,
            "details": [],
            "skipped": True,
            "reason": "no_ground_truth_medicines"
        }

    matched_count = 0
    matched_pred_indices = set()
    details = []

    for gt_med in gt_meds:
        gt_name = gt_med.get("name", "")
        found = False

        for pred_idx, pred_med in enumerate(pred_meds):
            if pred_idx in matched_pred_indices:
                continue

            pred_name = pred_med.get("name", "")
            if medicine_name_match(pred_name, gt_name):
                found = True
                matched_count += 1
                matched_pred_indices.add(pred_idx)
                break

        details.append({
            "gt_name": gt_name,
            "matched": found
        })

    score = matched_count / len(gt_meds) if gt_meds else 0.0

    return {
        "score": score,
        "matched": matched_count,
        "total_gt": len(gt_meds),
        "details": details,
        "skipped": False
    }


def dosage_integrity_score(pred_meds, gt_meds):
    total = 0
    correct = 0
    details = []

    for gt_med in gt_meds:
        gt_name = gt_med.get("name", "")
        gt_dose = normalize_text(gt_med.get("dose", ""))

        if not gt_dose:
            details.append({
                "medicine": gt_name,
                "status": "skipped",
                "reason": "dose_missing_in_ground_truth"
            })
            continue

        total += 1

        matched_pred = None
        for pred_med in pred_meds:
            if medicine_name_match(pred_med.get("name", ""), gt_name):
                matched_pred = pred_med
                break

        if matched_pred is None:
            details.append({
                "medicine": gt_name,
                "status": "wrong",
                "reason": "medicine_not_found",
                "gt_dose": gt_dose,
                "pred_dose": ""
            })
            continue

        pred_dose = normalize_text(matched_pred.get("dose", ""))
        is_correct = pred_dose == gt_dose

        if is_correct:
            correct += 1
            status = "correct"
        else:
            status = "wrong"

        details.append({
            "medicine": gt_name,
            "status": status,
            "gt_dose": gt_dose,
            "pred_dose": pred_dose
        })

    score = correct / total if total > 0 else 0.0

    return {
        "score": score,
        "correct": correct,
        "total": total,
        "details": details
    }


def spurious_detection_ratio(pred_meds, gt_meds):
    total_pred = len(pred_meds)

    if total_pred == 0:
        return {
            "score": 0.0,
            "false_positives": 0,
            "total_pred": 0,
            "details": []
        }

    false_positives = 0
    details = []

    for pred_med in pred_meds:
        pred_name = pred_med.get("name", "")
        matched = False

        for gt_med in gt_meds:
            gt_name = gt_med.get("name", "")
            if medicine_name_match(pred_name, gt_name):
                matched = True
                break

        if not matched:
            false_positives += 1

        details.append({
            "pred_name": pred_name,
            "is_spurious": not matched
        })

    score = false_positives / total_pred if total_pred > 0 else 0.0

    return {
        "score": score,
        "false_positives": false_positives,
        "total_pred": total_pred,
        "details": details
    }


# =========================================================
# Metric status helper
# =========================================================
def metric_status(metric_name, value):
    thresholds = {
        "extraction_coverage": 0.85,
        "structured_field_reliability": 0.85,
        "dosage_integrity_score": 0.90,
        "spurious_detection_ratio": 0.05
    }

    if metric_name not in thresholds:
        return "unknown"

    target = thresholds[metric_name]

    if metric_name == "spurious_detection_ratio":
        return "pass" if value <= target else "fail"

    return "pass" if value >= target else "fail"


# =========================================================
# Main evaluator
# =========================================================
def evaluate_prediction(pred, gt):
    pred = deepcopy(pred or {})
    gt = deepcopy(gt or {})

    pred_meds = normalize_medications(pred.get("medication", pred.get("medications", [])))
    gt_meds = normalize_medications(gt.get("medication", gt.get("medications", [])))

    main_struct = structured_field_reliability(pred, gt)
    vitals_cmp = compare_nested_dict(pred.get("vitals", {}), gt.get("vitals", {}), "vitals")
    followup_cmp = compare_nested_dict(pred.get("followup", {}), gt.get("followup", {}), "followup")

    structured_correct = main_struct["correct"] + vitals_cmp["correct"] + followup_cmp["correct"]
    structured_total = main_struct["total"] + vitals_cmp["total"] + followup_cmp["total"]
    structured_score = structured_correct / structured_total if structured_total > 0 else 0.0

    coverage = extraction_coverage(pred_meds, gt_meds)
    dosage = dosage_integrity_score(pred_meds, gt_meds)
    spurious = spurious_detection_ratio(pred_meds, gt_meds)

    metrics = {
        "extraction_coverage": round(coverage["score"], 4),
        "structured_field_reliability": round(structured_score, 4),
        "dosage_integrity_score": round(dosage["score"], 4),
        "spurious_detection_ratio": round(spurious["score"], 4)
    }

    targets = {
        "extraction_coverage_target": ">= 0.85",
        "structured_field_reliability_target": ">= 0.85",
        "dosage_integrity_score_target": ">= 0.90",
        "spurious_detection_ratio_target": "<= 0.05"
    }

    diagnostics = {
        "structured_fields": {
            "correct": structured_correct,
            "total": structured_total,
            "details": main_struct["details"] + vitals_cmp["details"] + followup_cmp["details"]
        },
        "medicine_coverage": coverage,
        "dosage_integrity": dosage,
        "spurious_detections": spurious,
        "evaluation_note": (
            "Only fields present in ground truth are evaluated. "
            "Fields empty or missing in ground truth are skipped and do not reduce the score."
        )
    }

    return metrics, targets, diagnostics


# =========================================================
# Wrapper
# =========================================================
def benchmark_json(prediction_json, ground_truth_json):
    if isinstance(prediction_json, str):
        prediction_json = json.loads(prediction_json)

    if isinstance(ground_truth_json, str):
        ground_truth_json = json.loads(ground_truth_json)

    metrics, targets, diagnostics = evaluate_prediction(prediction_json, ground_truth_json)

    return {
        "metrics": metrics,
        "targets": targets,
        "diagnostics": diagnostics
    }
