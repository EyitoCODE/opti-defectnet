"""Fail/defect is the positive class; undefined ratios are None."""


def classification_metrics(labels, predictions):
    if not labels or len(labels) != len(predictions):
        raise ValueError("Expected equally sized, nonempty labels and predictions")
    matrix = [[0, 0], [0, 0]]
    for actual, predicted in zip(labels, predictions):
        if actual not in (0, 1) or predicted not in (0, 1):
            raise ValueError("Labels must be 0 (pass) or 1 (defect)")
        matrix[actual][predicted] += 1
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    return {"samples": len(labels), "accuracy": (tn + tp) / len(labels),
            "defect_precision": tp / (tp + fp) if tp + fp else None,
            "defect_recall": tp / (tp + fn) if tp + fn else None,
            "defect_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "false_negatives": fn, "confusion_matrix": matrix,
            "class_order": ["pass", "fail"],
            "confusion_matrix_axes": "rows=actual, columns=predicted"}
