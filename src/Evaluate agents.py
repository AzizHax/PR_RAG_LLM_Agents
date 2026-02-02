#!/usr/bin/env python3
"""
Evaluation Script: Benchmark Agent 1 & Agent 2 against Ground Truth
- Load annotations and predictions
- Compute metrics (Acc, Prec, Rec, F1, AUC)
- Confusion matrix and error analysis
- Prepare for Agent 3 grid search
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

try:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score, classification_report
    )
except ImportError:
    print("ERROR: Please install scikit-learn: pip install scikit-learn numpy")
    exit(1)


# ============================================================================
# GROUND TRUTH LOADER
# ============================================================================

class GroundTruthLoader:
    """Load and parse ground truth annotations"""
    
    @staticmethod
    def load(annotations_path: str) -> Dict[str, int]:
        """
        Load ground truth from phantom_annotations_minimal.txt
        
        Format: PATIENT_ID: 0001 | LABEL_BINARY: 1 | COMMENT: ...
        
        Returns: {patient_id: label} where label is 0 (RA−) or 1 (RA+)
        """
        ground_truth = {}
        
        with open(annotations_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse line
                match = re.search(r'PATIENT_ID:\s*(\d+)\s*\|\s*LABEL_BINARY:\s*([01])', line)
                if match:
                    patient_id = match.group(1).zfill(4)  # Normalize to 4 digits
                    label = int(match.group(2))
                    ground_truth[patient_id] = label
        
        return ground_truth
    
    @staticmethod
    def get_stats(ground_truth: Dict[str, int]) -> Dict[str, int]:
        """Get label distribution statistics"""
        stats = {
            'total': len(ground_truth),
            'ra_positive': sum(1 for label in ground_truth.values() if label == 1),
            'ra_negative': sum(1 for label in ground_truth.values() if label == 0),
        }
        stats['prevalence'] = stats['ra_positive'] / stats['total'] if stats['total'] > 0 else 0
        return stats


# ============================================================================
# AGENT 1 EVALUATOR (Extraction + RAG)
# ============================================================================

@dataclass
class Agent1Scores:
    """Scores from Agent 1 extraction and RAG"""
    patient_id: str
    extraction_score: float  # Computed from facts
    rag_score_max: float
    rag_score_sum_top3: float
    rag_score_mean: float
    has_disease_mention: bool
    has_positive_serology: bool
    has_dmard_treatment: bool
    n_stays: int


class Agent1Evaluator:
    """Evaluate Agent 1 extraction quality and compute scores"""
    
    @staticmethod
    def compute_extraction_score(patient_facts: Dict) -> float:
        """
        Compute extraction score from patient facts
        Similar to your previous extract_score_v3
        """
        score = 0.0
        
        disease_mentions = patient_facts.get('disease_mentions', [])
        labs = patient_facts.get('labs', [])
        drugs = patient_facts.get('drugs', [])
        
        # Disease mention scoring
        confirmed_mentions = [m for m in disease_mentions if m.get('status') == 'confirmed']
        negated_mentions = [m for m in disease_mentions if m.get('status') == 'negated']
        
        if confirmed_mentions:
            score += 2.0
        elif disease_mentions and not negated_mentions:
            score += 1.0
        
        if negated_mentions:
            score -= 2.0
        
        # Serology scoring (with None handling)
        rf_positive = False
        acpa_positive = False
        
        for lab in labs:
            test = lab.get('test', '')
            if test is None:
                test = ''
            test = test.lower()
            
            polarity = lab.get('polarity', '')
            if polarity is None:
                polarity = ''
            polarity = polarity.lower()
            
            if test in ['rf', 'facteur rhumatoïde'] and polarity in ['positive', 'positif']:
                rf_positive = True
            
            if ('ccp' in test or 'acpa' in test) and polarity in ['positive', 'positif']:
                acpa_positive = True
        
        if rf_positive and acpa_positive:
            score += 3.0  # Double positive
        elif rf_positive or acpa_positive:
            score += 2.0  # Single positive
        
        # DMARD scoring (with None handling)
        has_mtx = False
        has_biologic = False
        has_jak = False
        
        for drug in drugs:
            name = drug.get('name', '')
            if name is None:
                name = ''
            name = name.lower()
            
            category = drug.get('category', '')
            if category is None:
                category = ''
            
            if 'methotrexate' in name or 'mtx' in name:
                has_mtx = True
            
            if category == 'bDMARD' or any(bio in name for bio in ['adalimumab', 'etanercept', 'infliximab', 'tocilizumab']):
                has_biologic = True
            
            if category == 'tsDMARD':
                has_jak = True
        
        if has_biologic or has_jak:
            score += 2.5
        elif has_mtx:
            score += 1.5
        
        return score
    
    @classmethod
    def evaluate(cls, patient_facts_path: str) -> Dict[str, Agent1Scores]:
        """Evaluate Agent 1 outputs and compute scores per patient"""
        
        with open(patient_facts_path, 'r', encoding='utf-8') as f:
            patients = [json.loads(line) for line in f]
        
        patient_scores = {}
        
        for patient in patients:
            patient_id = patient['patient_id']
            
            # Compute extraction score
            extraction_score = cls.compute_extraction_score(patient)
            
            # Get RAG metadata
            rag_meta = patient.get('_rag_meta_patient', {})
            rag_score_max = rag_meta.get('rag_score_max', 0.0)
            rag_score_sum_top3 = rag_meta.get('rag_score_sum_top3_max', 0.0)
            rag_score_mean = rag_meta.get('rag_score_mean', 0.0)
            
            # Binary flags (with None handling)
            has_disease = len(patient.get('disease_mentions', [])) > 0
            
            has_serology = False
            for lab in patient.get('labs', []):
                polarity = lab.get('polarity', '')
                if polarity is None:
                    polarity = ''
                if polarity.lower() in ['positive', 'positif']:
                    has_serology = True
                    break
            
            has_dmard = len(patient.get('drugs', [])) > 0
            
            patient_scores[patient_id] = Agent1Scores(
                patient_id=patient_id,
                extraction_score=extraction_score,
                rag_score_max=rag_score_max,
                rag_score_sum_top3=rag_score_sum_top3,
                rag_score_mean=rag_score_mean,
                has_disease_mention=has_disease,
                has_positive_serology=has_serology,
                has_dmard_treatment=has_dmard,
                n_stays=patient.get('n_stays', 0)
            )
        
        return patient_scores


# ============================================================================
# AGENT 2 EVALUATOR (Clinical Reasoning)
# ============================================================================

class Agent2Evaluator:
    """Evaluate Agent 2 criteria assessment predictions"""
    
    @staticmethod
    def load_predictions(criteria_dir: str) -> Dict[str, Dict]:
        """
        Load Agent 2 predictions from criteria_assessment_XXXX.json files
        
        Args:
            criteria_dir: Directory containing criteria_assessment_XXXX.json files
        
        Returns:
            Dict[patient_id, assessment_dict]
        """
        criteria_dir = Path(criteria_dir)
        
        if not criteria_dir.exists():
            raise FileNotFoundError(f"Criteria directory not found: {criteria_dir}")
        
        pred_dict = {}
        
        # Load all criteria_assessment_*.json files
        for assessment_file in sorted(criteria_dir.glob('criteria_assessment_*.json')):
            with open(assessment_file, 'r', encoding='utf-8') as f:
                assessment = json.load(f)
            
            patient_id = assessment.get('patient_id')
            if patient_id:
                pred_dict[patient_id] = assessment
        
        return pred_dict
    
    @staticmethod
    def convert_label_to_binary(label: str, uncertain_as_negative: bool = True) -> int:
        """
        Convert Agent 2 label to binary (0/1)
        
        Agent 2 output format: final_label = "RA+" or "RA−"
        """
        if label == "RA+":
            return 1
        elif label == "RA−":
            return 0
        else:  # Handle any edge cases
            return 0 if uncertain_as_negative else None


# ============================================================================
# METRICS CALCULATOR
# ============================================================================

class MetricsCalculator:
    """Calculate classification metrics"""
    
    @staticmethod
    def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """Compute standard classification metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['tp'] = int(tp)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        metrics['tn'] = int(tn)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return metrics
    
    @staticmethod
    def compute_threshold_metrics(y_true: List[int], scores: List[float], threshold: float) -> Dict[str, float]:
        """Compute metrics for a given threshold on continuous scores"""
        y_pred = [1 if s >= threshold else 0 for s in scores]
        return MetricsCalculator.compute_metrics(y_true, y_pred)
    
    @staticmethod
    def grid_search_threshold(y_true: List[int], scores: List[float], 
                             thresholds: List[float]) -> Tuple[float, Dict[str, float]]:
        """
        Grid search for best threshold on scores
        Returns: (best_threshold, best_metrics)
        """
        best_f1 = -1
        best_threshold = None
        best_metrics = None
        
        for threshold in thresholds:
            metrics = MetricsCalculator.compute_threshold_metrics(y_true, scores, threshold)
            
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_threshold = threshold
                best_metrics = metrics
        
        return best_threshold, best_metrics


# ============================================================================
# MAIN EVALUATION
# ============================================================================

class MainEvaluator:
    """Main evaluation orchestrator"""
    
    def __init__(self, 
                 annotations_path: str,
                 patient_facts_path: str,
                 criteria_dir: str):
        
        self.annotations_path = annotations_path
        self.patient_facts_path = patient_facts_path
        self.criteria_dir = criteria_dir
        
        # Load data
        print("[Evaluator] Loading ground truth annotations...")
        self.ground_truth = GroundTruthLoader.load(annotations_path)
        
        print("[Evaluator] Loading Agent 1 patient facts...")
        self.agent1_scores = Agent1Evaluator.evaluate(patient_facts_path)
        
        print("[Evaluator] Loading Agent 2 criteria assessments...")
        self.agent2_predictions = Agent2Evaluator.load_predictions(criteria_dir)
        
        print(f"[Evaluator] ✓ Loaded {len(self.ground_truth)} ground truth labels")
        print(f"[Evaluator] ✓ Loaded {len(self.agent1_scores)} Agent 1 scores")
        print(f"[Evaluator] ✓ Loaded {len(self.agent2_predictions)} Agent 2 predictions\n")
    
    def evaluate_agent2(self, uncertain_as_negative: bool = True) -> Dict:
        """Evaluate Agent 2 predictions from criteria_assessment files"""
        
        y_true = []
        y_pred = []
        patient_ids = []
        
        for patient_id in sorted(self.ground_truth.keys()):
            if patient_id not in self.agent2_predictions:
                print(f"WARNING: Patient {patient_id} not in Agent 2 predictions, skipping")
                continue
            
            true_label = self.ground_truth[patient_id]
            
            # Get final_label from criteria_assessment
            pred_label_str = self.agent2_predictions[patient_id].get('final_label', 'RA−')
            pred_label = Agent2Evaluator.convert_label_to_binary(pred_label_str, uncertain_as_negative)
            
            if pred_label is None:
                continue
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            patient_ids.append(patient_id)
        
        metrics = MetricsCalculator.compute_metrics(y_true, y_pred)
        
        # Count uncertain cases (though shouldn't exist in current Agent 2)
        n_uncertain = sum(
            1 for p in self.agent2_predictions.values() 
            if p.get('final_label', 'RA−') not in ['RA+', 'RA−']
        )
        
        return {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred': y_pred,
            'patient_ids': patient_ids,
            'n_uncertain': n_uncertain
        }
    
    def evaluate_agent1_with_threshold(self, score_type: str = 'extraction', 
                                       thresholds: List[float] = None) -> Dict:
        """
        Evaluate Agent 1 scores with threshold grid search
        
        score_type: 'extraction', 'rag_max', 'rag_sum', 'combined'
        """
        
        if thresholds is None:
            thresholds = np.arange(0.0, 10.0, 0.25)
        
        y_true = []
        scores = []
        patient_ids = []
        
        for patient_id in sorted(self.ground_truth.keys()):
            if patient_id not in self.agent1_scores:
                print(f"WARNING: Patient {patient_id} not in Agent 1 scores, skipping")
                continue
            
            true_label = self.ground_truth[patient_id]
            agent1_score = self.agent1_scores[patient_id]
            
            # Select score type
            if score_type == 'extraction':
                score = agent1_score.extraction_score
            elif score_type == 'rag_max':
                score = agent1_score.rag_score_max
            elif score_type == 'rag_sum':
                score = agent1_score.rag_score_sum_top3
            elif score_type == 'combined':
                score = max(agent1_score.extraction_score, agent1_score.rag_score_max)
            else:
                raise ValueError(f"Unknown score_type: {score_type}")
            
            y_true.append(true_label)
            scores.append(score)
            patient_ids.append(patient_id)
        
        # Grid search
        best_threshold, best_metrics = MetricsCalculator.grid_search_threshold(
            y_true, scores, thresholds
        )
        
        return {
            'score_type': score_type,
            'best_threshold': best_threshold,
            'best_metrics': best_metrics,
            'y_true': y_true,
            'scores': scores,
            'patient_ids': patient_ids
        }
    
    def compare_all_approaches(self):
        """Compare Agent 1, Agent 2, and combinations"""
        
        print("=" * 80)
        print("EVALUATION REPORT: Agent 1 vs Agent 2 vs Ground Truth")
        print("=" * 80)
        
        # Ground truth stats
        gt_stats = GroundTruthLoader.get_stats(self.ground_truth)
        print(f"\nGround Truth Statistics:")
        print(f"  Total patients: {gt_stats['total']}")
        print(f"  RA+ (positive): {gt_stats['ra_positive']} ({100*gt_stats['prevalence']:.1f}%)")
        print(f"  RA− (negative): {gt_stats['ra_negative']} ({100*(1-gt_stats['prevalence']):.1f}%)")
        
        # Agent 2 evaluation
        print("\n" + "-" * 80)
        print("AGENT 2: Clinical Reasoning (ACR-EULAR + DMARD Logic)")
        print("-" * 80)
        
        agent2_results = self.evaluate_agent2(uncertain_as_negative=True)
        self._print_metrics(agent2_results['metrics'])
        print(f"\nUncertain cases (treated as RA−): {agent2_results['n_uncertain']}")
        
        # Agent 1 evaluation (extraction score)
        print("\n" + "-" * 80)
        print("AGENT 1: Extraction Score (with threshold optimization)")
        print("-" * 80)
        
        agent1_extraction = self.evaluate_agent1_with_threshold(score_type='extraction')
        print(f"Best threshold: {agent1_extraction['best_threshold']:.2f}")
        self._print_metrics(agent1_extraction['best_metrics'])
        
        # Agent 1 evaluation (RAG score)
        print("\n" + "-" * 80)
        print("AGENT 1: RAG Score Max (with threshold optimization)")
        print("-" * 80)
        
        agent1_rag = self.evaluate_agent1_with_threshold(score_type='rag_max')
        print(f"Best threshold: {agent1_rag['best_threshold']:.2f}")
        self._print_metrics(agent1_rag['best_metrics'])
        
        # Agent 1 evaluation (combined)
        print("\n" + "-" * 80)
        print("AGENT 1: Combined Score (max of extraction & RAG)")
        print("-" * 80)
        
        agent1_combined = self.evaluate_agent1_with_threshold(score_type='combined')
        print(f"Best threshold: {agent1_combined['best_threshold']:.2f}")
        self._print_metrics(agent1_combined['best_metrics'])
        
        # Summary comparison
        print("\n" + "=" * 80)
        print("SUMMARY COMPARISON")
        print("=" * 80)
        
        comparison = {
            'Agent 2 (Reasoning)': agent2_results['metrics'],
            'Agent 1 (Extraction)': agent1_extraction['best_metrics'],
            'Agent 1 (RAG)': agent1_rag['best_metrics'],
            'Agent 1 (Combined)': agent1_combined['best_metrics'],
        }
        
        self._print_comparison_table(comparison)
        
        # Error analysis
        print("\n" + "=" * 80)
        print("ERROR ANALYSIS: Agent 2")
        print("=" * 80)
        self._error_analysis(agent2_results)
        
        return {
            'agent2': agent2_results,
            'agent1_extraction': agent1_extraction,
            'agent1_rag': agent1_rag,
            'agent1_combined': agent1_combined
        }
    
    def _print_metrics(self, metrics: Dict[str, float]):
        """Print metrics in formatted way"""
        print(f"  Accuracy:    {metrics['accuracy']:.3f}")
        print(f"  Precision:   {metrics['precision']:.3f}")
        print(f"  Recall:      {metrics['recall']:.3f}")
        print(f"  F1-score:    {metrics['f1']:.3f}")
        print(f"  Specificity: {metrics['specificity']:.3f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TP={metrics['tp']}  FP={metrics['fp']}")
        print(f"    FN={metrics['fn']}  TN={metrics['tn']}")
    
    def _print_comparison_table(self, comparison: Dict[str, Dict]):
        """Print comparison table across approaches"""
        print(f"\n{'Approach':<25} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
        print("-" * 65)
        for name, metrics in comparison.items():
            print(f"{name:<25} {metrics['accuracy']:<8.3f} {metrics['precision']:<8.3f} "
                  f"{metrics['recall']:<8.3f} {metrics['f1']:<8.3f}")
    
    def _error_analysis(self, results: Dict):
        """Detailed error analysis"""
        y_true = results['y_true']
        y_pred = results['y_pred']
        patient_ids = results['patient_ids']
        
        # False positives
        fp_patients = [pid for pid, yt, yp in zip(patient_ids, y_true, y_pred) 
                       if yt == 0 and yp == 1]
        
        # False negatives
        fn_patients = [pid for pid, yt, yp in zip(patient_ids, y_true, y_pred) 
                       if yt == 1 and yp == 0]
        
        print(f"\nFalse Positives (predicted RA+ but actually RA−): {len(fp_patients)}")
        for pid in fp_patients:
            pred = self.agent2_predictions[pid]
            # Handle both 'justification' (old) and 'classification_rationale' (new LLM)
            justif = pred.get('classification_rationale') or pred.get('justification', 'No justification available')
            print(f"  - Patient {pid}: {justif[:80]}...")
        
        print(f"\nFalse Negatives (predicted RA− but actually RA+): {len(fn_patients)}")
        for pid in fn_patients:
            pred = self.agent2_predictions[pid]
            justif = pred.get('classification_rationale') or pred.get('justification', 'No justification available')
            print(f"  - Patient {pid}: {justif[:80]}...")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution with hardcoded Windows paths"""
    
    # Hardcoded paths
    ANNOTATIONS_PATH = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data\phantom_annotations_minimal.txt"
    PATIENT_FACTS_PATH = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data\facts_agent1_patient.jsonl"
    CRITERIA_DIR = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data\criteria_assessments_llm"
    OUTPUT_PATH = r"C:\Users\Mega-Pc\Desktop\Projects\PhenoRag\fantôme\Data\evaluation_results.json"
    
    # Check files exist
    for path in [ANNOTATIONS_PATH, PATIENT_FACTS_PATH]:
        if not Path(path).exists():
            print(f"ERROR: File not found: {path}")
            return
    
    if not Path(CRITERIA_DIR).exists():
        print(f"ERROR: Directory not found: {CRITERIA_DIR}")
        return
    
    # Run evaluation
    evaluator = MainEvaluator(
        annotations_path=ANNOTATIONS_PATH,
        patient_facts_path=PATIENT_FACTS_PATH,
        criteria_dir=CRITERIA_DIR
    )
    
    results = evaluator.compare_all_approaches()
    
    # Save results to JSON
    # Convert numpy types to Python types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        return obj
    
    results_serializable = convert_to_json_serializable(results)
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Detailed results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()