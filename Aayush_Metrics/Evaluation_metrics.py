"""
evaluate_metrics.py
--------------------
Computes 8 evaluation metrics by comparing model output JSON
against ground truth label JSONs.

Run AFTER evaluate.py has generated results_TIMESTAMP.json

Usage:
    python evaluate_metrics.py --results output/results_20260325_151749.json

Ground truth label files must be in data/labels/ with the same
stem as the image filename  e.g. 1.json for 1.jpg
"""

import os
import re
import json
import argparse
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

LABELS_DIR = 'data/labels'
OUTPUT_DIR = 'output'

HIGH_ALERT = {
    'digoxin', 'warfarin', 'insulin', 'methotrexate', 'lithium',
    'heparin', 'phenytoin', 'clexane', 'enoxaparin', 'fentanyl',
    'morphine', 'amiodarone', 'vancomycin',
}

TARGETS = {
    'medicine_recall':        ('>=', 0.80),
    'medicine_precision':     ('>=', 0.85),
    'field_accuracy_doctor':  ('>=', 0.85),
    'field_accuracy_patient': ('>=', 0.85),
    'field_accuracy_date':    ('>=', 0.85),
    'false_positive_rate':    ('<=', 0.05),
    'dose_accuracy':          ('>=', 0.90),
    'critical_failure_rate':  ('==', 0.00),
    'pipeline_failure_rate':  ('<=', 0.02),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

PREFIX_RE = re.compile(
    r'^(?:tab\.?|tablet\.?|cap\.?|capsule\.?|syp\.?|syr\.?|syrup\.?|'
    r'inj\.?|injection\.?|drops?\.?|oint\.?|cream\.?|t\.)\s+',
    re.IGNORECASE
)

DOSE_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*(mg|mcg|ml|g|iu|units?)',
    re.IGNORECASE
)


def normalise_name(name: str) -> str:
    name = PREFIX_RE.sub('', name.strip())
    name = re.sub(r'\s*\(.*?\)', '', name)
    name = re.sub(
        r'\s+\d+(?:\.\d+)?\s*(?:mg|mcg|ml|g|iu|units?).*$',
        '', name, flags=re.IGNORECASE
    )
    return name.lower().strip()


def names_match(a: str, b: str) -> bool:
    na, nb = normalise_name(a), normalise_name(b)
    if not na or not nb:
        return False
    return na in nb or nb in na


def parse_dose(dose_str: str):
    if not dose_str:
        return None
    m = DOSE_RE.search(dose_str)
    if not m:
        return None
    return float(m.group(1)), m.group(2).lower().rstrip('s')


def doses_match(a: str, b: str) -> bool:
    pa, pb = parse_dose(a), parse_dose(b)
    if not pa or not pb:
        return False
    va, ua = pa
    vb, ub = pb
    if ua != ub:
        return False
    return abs(va - vb) / max(va, vb) < 0.10


def is_critical_failure(name: str, gt_dose: str, pred_dose: str) -> bool:
    if normalise_name(name) not in HIGH_ALERT:
        return False
    pa, pb = parse_dose(gt_dose), parse_dose(pred_dose)
    if not pa or not pb:
        return False
    va, ua = pa
    vb, ub = pb
    if ua != ub or va == 0:
        return False
    ratio = vb / va
    return ratio >= 5.0 or ratio <= 0.2


def field_match(gt: str, pred: str) -> bool:
    if not gt or not pred:
        return False
    return gt.lower().strip() in pred.lower().strip() or \
           pred.lower().strip() in gt.lower().strip()


# ── Ground truth loader ───────────────────────────────────────────────────────

def load_ground_truth(label_path: Path) -> dict:
    """
    Handles two label formats:
      Format 1 — 'prescription' key  (used by your current evaluate.py)
      Format 2 — 'medication' key    (full label JSON structure)
    """
    with open(label_path, encoding='utf-8') as f:
        raw = json.load(f)

    medicines = []

    if 'prescription' in raw:
        for entry in raw.get('prescription', []):
            name = entry.get('name', '').strip()
            if name:
                medicines.append({
                    'name':      name,
                    'dose':      entry.get('dose', '').strip(),
                    'frequency': entry.get('freq', '').strip(),
                    'route':     entry.get('route', '').strip(),
                })

    elif 'medication' in raw:
        for med in raw.get('medication', []):
            name = med.get('name', '').strip()
            if not name:
                continue
            details = med.get('medicationDetails') or [{}]
            d = details[0] if details else {}
            medicines.append({
                'name':      name,
                'dose':      d.get('dose', '').strip(),
                'frequency': d.get('freq', '').strip(),
                'route':     d.get('route', '').strip(),
            })

    return {
        'doctor':    raw.get('doctorUsername', '').strip(),
        'patient':   raw.get('name', '').strip(),
        'date':      raw.get('date', '').strip(),
        'medicines': medicines,
    }


# ── Per-image evaluation ──────────────────────────────────────────────────────

def evaluate_image(model_out: dict, gt: dict) -> dict:
    pred_meds = model_out.get('medicines', [])
    gt_meds   = gt['medicines']

    row = {
        'image':             model_out.get('image', ''),
        'gt_count':          len(gt_meds),
        'pred_count':        len(pred_meds),
        'found':             0,
        'false_positives':   0,
        'doctor_gt':         gt['doctor'],
        'patient_gt':        gt['patient'],
        'date_gt':           gt['date'],
        'doctor_pred':       model_out.get('doctor', ''),
        'patient_pred':      model_out.get('patient', ''),
        'date_pred':         model_out.get('date', ''),
        'doctor_match':      False,
        'patient_match':     False,
        'date_match':        False,
        'doses_evaluated':   0,
        'doses_correct':     0,
        'critical_failures': [],
        'recall':            None,
        'precision':         None,
    }

    if gt['doctor']:
        row['doctor_match']  = field_match(gt['doctor'],  row['doctor_pred'])
    if gt['patient']:
        row['patient_match'] = field_match(gt['patient'], row['patient_pred'])
    if gt['date']:
        row['date_match']    = field_match(gt['date'],    row['date_pred'])

    matched_pred = set()
    for gt_med in gt_meds:
        for j, pm in enumerate(pred_meds):
            if j in matched_pred:
                continue
            if names_match(gt_med['name'], pm.get('name', '')):
                row['found'] += 1
                matched_pred.add(j)
                gt_dose   = gt_med.get('dose', '')
                pred_dose = pm.get('dose', '')
                if gt_dose:
                    row['doses_evaluated'] += 1
                    if doses_match(gt_dose, pred_dose):
                        row['doses_correct'] += 1
                    if is_critical_failure(gt_med['name'], gt_dose, pred_dose):
                        row['critical_failures'].append({
                            'medicine':  gt_med['name'],
                            'gt_dose':   gt_dose,
                            'pred_dose': pred_dose,
                        })
                break

    row['false_positives'] = sum(
        1 for j in range(len(pred_meds)) if j not in matched_pred
    )

    if gt_meds:
        row['recall']    = row['found'] / len(gt_meds)
    if pred_meds:
        row['precision'] = (len(pred_meds) - row['false_positives']) / len(pred_meds)

    return row


# ── Aggregate ─────────────────────────────────────────────────────────────────

def aggregate(per_image: list, total_images: int, failed_images: int) -> dict:

    def avg(vals):
        return sum(vals) / len(vals) if vals else 0.0

    def field_acc(match_key, gt_key):
        eligible = [r for r in per_image if r[gt_key]]
        return avg([1 if r[match_key] else 0 for r in eligible]) if eligible else None

    recall_vals = [r['recall']    for r in per_image if r['recall']    is not None]
    prec_vals   = [r['precision'] for r in per_image if r['precision'] is not None]
    avg_recall  = avg(recall_vals)
    avg_prec    = avg(prec_vals)
    f1 = (2 * avg_recall * avg_prec / (avg_recall + avg_prec)
          if (avg_recall + avg_prec) > 0 else 0.0)

    total_pred    = sum(r['pred_count']      for r in per_image)
    total_fp      = sum(r['false_positives'] for r in per_image)
    total_doses   = sum(r['doses_evaluated'] for r in per_image)
    correct_doses = sum(r['doses_correct']   for r in per_image)
    all_critical  = [cf for r in per_image for cf in r['critical_failures']]

    return {
        'medicine_recall':        round(avg_recall, 4),
        'medicine_precision':     round(avg_prec,   4),
        'medicine_f1':            round(f1,         4),
        'field_accuracy_doctor':  round(field_acc('doctor_match',  'doctor_gt'),  4) if field_acc('doctor_match',  'doctor_gt')  is not None else None,
        'field_accuracy_patient': round(field_acc('patient_match', 'patient_gt'), 4) if field_acc('patient_match', 'patient_gt') is not None else None,
        'field_accuracy_date':    round(field_acc('date_match',    'date_gt'),    4) if field_acc('date_match',    'date_gt')    is not None else None,
        'false_positive_rate':    round(total_fp / total_pred, 4) if total_pred > 0 else 0.0,
        'dose_accuracy':          round(correct_doses / total_doses, 4) if total_doses > 0 else None,
        'critical_failure_rate':  round(len(all_critical) / max(total_pred, 1), 6),
        'critical_failures':      all_critical,
        'pipeline_failure_rate':  round(failed_images / total_images, 4) if total_images > 0 else 0.0,
        'total_images':           total_images,
        'failed_images':          failed_images,
        'images_with_gt':         len(per_image),
        'total_gt_medicines':     sum(r['gt_count']         for r in per_image),
        'total_pred_medicines':   total_pred,
        'medicines_found':        sum(r['found']            for r in per_image),
        'total_false_positives':  total_fp,
        'total_doses_evaluated':  total_doses,
        'total_doses_correct':    correct_doses,
    }


def pass_fail(metrics: dict) -> dict:
    results = {}
    for metric, (op, target) in TARGETS.items():
        val = metrics.get(metric)
        if val is None:
            results[metric] = 'NO DATA'
            continue
        if op == '>=':
            results[metric] = 'PASS' if val >= target else 'FAIL'
        elif op == '<=':
            results[metric] = 'PASS' if val <= target else 'FAIL'
        elif op == '==':
            results[metric] = 'PASS' if val == target else 'FAIL'
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def run(results_path: str):
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    with open(results_path, encoding='utf-8') as f:
        model_results = json.load(f)

    labels_dir    = Path(LABELS_DIR)
    per_image     = []
    skipped       = []
    failed_images = 0

    for model_out in model_results:
        image_name = model_out.get('image', '')
        stem       = Path(image_name).stem
        label_path = labels_dir / f'{stem}.json'

        if not label_path.exists():
            skipped.append(image_name)
            continue

        if (not model_out.get('medicines') and
            not model_out.get('doctor') and
            not model_out.get('patient')):
            failed_images += 1

        gt = load_ground_truth(label_path)
        per_image.append(evaluate_image(model_out, gt))

    total_images = len(model_results)
    metrics      = aggregate(per_image, total_images, failed_images)
    pf           = pass_fail(metrics)

    W = 65
    print(f"\n{'='*W}")
    print(f"  EVALUATION METRICS REPORT")
    print(f"{'='*W}")
    print(f"  Results file  : {results_path}")
    print(f"  Labels dir    : {LABELS_DIR}")
    print(f"  Total images  : {total_images}")
    print(f"  Matched to GT : {len(per_image)}")
    if skipped:
        print(f"  Skipped       : {skipped}")
    print(f"{'='*W}")

    rows = [
        ('Medicine Recall',       'medicine_recall',        '>= 0.80'),
        ('Medicine Precision',    'medicine_precision',     '>= 0.85'),
        ('Medicine F1',           'medicine_f1',            '(info)'),
        ('Field Acc — Doctor',    'field_accuracy_doctor',  '>= 0.85'),
        ('Field Acc — Patient',   'field_accuracy_patient', '>= 0.85'),
        ('Field Acc — Date',      'field_accuracy_date',    '>= 0.85'),
        ('False Positive Rate',   'false_positive_rate',    '<= 0.05'),
        ('Dose Accuracy',         'dose_accuracy',          '>= 0.90'),
        ('Critical Failure Rate', 'critical_failure_rate',  '== 0.00'),
        ('Pipeline Failure Rate', 'pipeline_failure_rate',  '<= 0.02'),
    ]

    print(f"\n  {'Metric':<26} {'Value':>8}  {'Target':<12}  Status")
    print(f"  {'-'*58}")
    for label, key, target_str in rows:
        val    = metrics.get(key)
        status = pf.get(key, '')
        v_str  = f'{val:.4f}' if isinstance(val, float) else str(val)
        print(f"  {label:<26} {v_str:>8}  {target_str:<12}  {status}")

    print(f"\n{'='*W}")
    print(f"  COUNTS")
    print(f"{'='*W}")
    print(f"  GT medicines total     : {metrics['total_gt_medicines']}")
    print(f"  Predicted medicines    : {metrics['total_pred_medicines']}")
    print(f"  Correctly matched      : {metrics['medicines_found']}")
    print(f"  False positives        : {metrics['total_false_positives']}")
    print(f"  Doses evaluated        : {metrics['total_doses_evaluated']}")
    print(f"  Doses correct          : {metrics['total_doses_correct']}")

    if metrics['critical_failures']:
        print(f"\n{'='*W}")
        print(f"  CRITICAL FAILURES ({len(metrics['critical_failures'])})")
        print(f"{'='*W}")
        for cf in metrics['critical_failures']:
            print(f"  Medicine : {cf['medicine']}")
            print(f"  GT dose  : {cf['gt_dose']}")
            print(f"  Pred dose: {cf['pred_dose']}")
    else:
        print(f"\n  No critical failures detected.")

    passes = sum(1 for v in pf.values() if v == 'PASS')
    fails  = sum(1 for v in pf.values() if v == 'FAIL')
    print(f"\n{'='*W}")
    print(f"  OVERALL: {passes} PASS  |  {fails} FAIL")
    print(f"{'='*W}")

    print(f"\n  PER IMAGE BREAKDOWN")
    print(f"  {'Image':<12} {'GT':>4} {'Pred':>5} {'Found':>6} {'FP':>4} {'Recall':>8}  Doc  Pat  Date")
    print(f"  {'-'*65}")
    for r in per_image:
        rec  = f"{r['recall']:.2f}" if r['recall'] is not None else 'N/A'
        doc  = 'Y' if r['doctor_match']  else ('N' if r['doctor_gt']  else '-')
        pat  = 'Y' if r['patient_match'] else ('N' if r['patient_gt'] else '-')
        date = 'Y' if r['date_match']    else ('N' if r['date_gt']    else '-')
        print(f"  {r['image']:<12} {r['gt_count']:>4} {r['pred_count']:>5} {r['found']:>6} {r['false_positives']:>4} {rec:>8}    {doc}    {pat}    {date}")

    ts          = Path(results_path).stem.replace('results_', '')
    report_path = f'{OUTPUT_DIR}/metrics_{ts}.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics':   metrics,
            'pass_fail': pf,
            'per_image': per_image,
            'skipped':   skipped,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Full report saved: {report_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results', required=True,
        help='Path to results JSON from evaluate.py'
    )
    args = parser.parse_args()
    run(args.results)
