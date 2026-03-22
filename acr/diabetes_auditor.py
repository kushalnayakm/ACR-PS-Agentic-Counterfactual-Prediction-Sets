import json
import pandas as pd
from diabetes_rules import DIABETES_RULES

class DiabetesAuditor:
    def __init__(self, rules):
        self.rules = rules

    def audit(self, query, cf_list):
        valid = []
        invalid = []
        
        for cf in cf_list:
            is_valid = True
            reason = ""
            
            for feature, val in cf.items():
                if feature == 'Outcome': continue
                
                rule = self.rules.get(feature)
                if not rule: continue
                
                # Check Mutability (Age/Genetics)
                if not rule['mutable'] and val != query[feature]:
                    is_valid = False
                    reason = f"Feature '{feature}' is immutable. Suggestion: {query[feature]} -> {val}"
                    break
                    
                # Check Directional Constraints (Pregnancies)
                if rule.get('constraint') == 'increase_only' and val < query[feature]:
                    is_valid = False
                    reason = f"Feature '{feature}' cannot decrease. Suggestion: {query[feature]} -> {val}"
                    break
                    
            if is_valid:
                valid.append(cf)
            else:
                invalid.append({"cf": cf, "reason": reason})
                
        return valid, invalid

def run_audit():
    with open("acr/diabetes/raw_cf.json", 'r') as f:
        data = json.load(f)
        
    query = data['test_data'][0][0]
    query_obj = dict(zip(data['feature_names'], query))
    cfs = [dict(zip(data['feature_names'], cf)) for cf in data['cfs_list'][0]]
    
    auditor = DiabetesAuditor(DIABETES_RULES)
    valid, invalid = auditor.audit(query_obj, cfs)
    
    # Save the audit report
    report = {
        "original_data": query_obj,
        "valid_counterfactuals": valid,
        "invalid_suggestions": invalid,
    }
    
    with open("acr/diabetes/filtered_cf.json", 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Step 3 Audit Complete:")
    print(f"  Total Raw Suggestions: {len(cfs)}")
    print(f"  Faithful (Actionable): {len(valid)}")
    print(f"  Faithless (Discarded): {len(invalid)}")

if __name__ == "__main__":
    run_audit()
