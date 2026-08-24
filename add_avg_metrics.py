import re
import sys

def main():
    new_func = """
def average_client_metrics(client_models, dataset_test, num_classes: int, buckets, device: str):
    all_scores = []
    
    from collections import defaultdict
    for m in client_models.values():
        cm = compute_confusion_matrix(m, dataset_test, num_classes, device)
        report = classification_report_from_cm(cm)
        
        recalls = [report[c]["recall"] for c in range(num_classes)]
        bucket_scores = defaultdict(list)
        for c, acc in enumerate(recalls):
            bucket_scores[buckets[c]].append(acc)
            
        scores = {b: sum(v) / max(len(v), 1) for b, v in bucket_scores.items()}
        scores["overall"] = report["class_balanced_accuracy"]
        scores["class_balanced_accuracy"] = report["class_balanced_accuracy"]
        scores["macro_f1"] = report["macro_avg"]["f1"]
        scores["weighted_f1"] = report["weighted_avg"]["f1"]
        scores["macro_precision"] = report["macro_avg"]["precision"]
        scores["macro_recall"] = report["macro_avg"]["recall"]
        
        all_scores.append(scores)
        
    keys = set().union(*[s.keys() for s in all_scores]) if all_scores else set()
    return {k: sum(s.get(k, 0.0) for s in all_scores) / len(all_scores) for k in keys}
"""
    with open('utils/evaluate.py', 'a', encoding='utf-8') as f:
        f.write(new_func)

if __name__ == '__main__':
    main()
