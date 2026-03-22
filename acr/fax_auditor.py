import json
import os
from acr.causal_rulebook import CAUSAL_RULES

# Education ranking for "increase_only" rule
EDUCATION_RANK = [
    'School', 'HS-grad', 'Some-college', 'Assoc', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate'
]

class FAXAuditor:
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
        self.rules = CAUSAL_RULES
        self.raw_cf_data = self._load_data()
        self.feature_names = self.raw_cf_data.get('feature_names_including_target', [])
        self.results = []

    def _load_data(self):
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Raw data not found at {self.raw_data_path}")
        with open(self.raw_data_path, 'r') as f:
            return json.load(f)

    def audit(self):
        print("\nStep 3: Auditing counterfactuals against Causal Rulebook...")
        
        test_data = self.raw_cf_data.get('test_data', [])
        cfs_list = self.raw_cf_data.get('cfs_list', [])
        
        for idx, (original, cfs) in enumerate(zip(test_data, cfs_list)):
            # original is a list of lists ([values])
            original_values = original[0]
            original_dict = dict(zip(self.feature_names, original_values))
            
            print(f"\nAnalyzing Sample {idx + 1}...")
            
            sample_results = {
                "sample_id": idx + 1,
                "original_data": original_dict,
                "valid_counterfactuals": [],
                "invalid_suggestions": []
            }
            
            for cf_values in cfs:
                cf_dict = dict(zip(self.feature_names, cf_values))
                is_valid, reason = self.filter_cf(original_dict, cf_dict)
                
                if is_valid:
                    sample_results["valid_counterfactuals"].append(cf_dict)
                else:
                    sample_results["invalid_suggestions"].append({
                        "suggestion": cf_dict,
                        "reason": reason
                    })
            
            self.results.append(sample_results)
            print(f" - Found {len(sample_results['valid_counterfactuals'])} valid CFs")
            print(f" - Discarded {len(sample_results['invalid_suggestions'])} invalid suggestions")

    def filter_cf(self, original_row, cf_row):
        for feature, new_val in cf_row.items():
            if feature == 'income': # Skip the target
                continue
                
            old_val = original_row.get(feature)
            if old_val == new_val:
                continue
            
            rule = self.rules.get(feature)
            if not rule:
                continue
                
            # Rule: Immutable
            if not rule.get('mutable', True):
                return False, f"Feature '{feature}' is immutable. Suggestion: {old_val} -> {new_val}"
            
            # Rule: Directional Constraint (e.g., Education)
            if rule.get('direction') == 'increase_only' and feature == 'education':
                try:
                    old_rank = EDUCATION_RANK.index(old_val)
                    new_rank = EDUCATION_RANK.index(new_val)
                    if new_rank < old_rank:
                        return False, f"Education cannot decrease: {old_val} -> {new_val}"
                except ValueError:
                    # If value not in rank list, ignore for baseline
                    pass
                
        return True, "Valid"

    def save_results(self, output_path="acr/filtered_counterfactuals.json"):
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"\nFiltered results saved to {output_path}")

if __name__ == "__main__":
    auditor = FAXAuditor("acr/raw_counterfactuals.json")
    auditor.audit()
    auditor.save_results()

