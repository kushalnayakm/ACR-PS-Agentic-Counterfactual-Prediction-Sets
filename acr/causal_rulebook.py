# ACR Causal Rulebook
# Defined in Project Reference Document Step 2

CAUSAL_RULES = {
    "age": {
        "mutable": False,
        "reason": "Biological - age cannot be reversed or skipped in a counterfactual sense."
    },
    "race": {
        "mutable": False,
        "reason": "Immutable characteristic."
    },
    "gender": {
        "mutable": False,
        "reason": "Immutable characteristic for the purpose of this baseline model."
    },
    "native-country": {
        "mutable": False,
        "reason": "Place of birth cannot be changed."
    },
    "education": {
        "mutable": True,
        "direction": "increase_only",
        "reason": "Education level can be attained but not lost."
    },
    "marital_status": {
        "mutable": True,
        "reason": "Marital status can change via legal processes."
    },
    "occupation": {
        "mutable": True,
        "reason": "A person can change their job/career."
    },
    "hours_per_week": {
        "mutable": True,
        "reason": "Work hours are generally negotiable or changeable."
    },
    "workclass": {
        "mutable": True,
        "reason": "Employment sector can change."
    }
}

def get_rule(feature_name):
    return CAUSAL_RULES.get(feature_name, {"mutable": True, "reason": "No specific constraint defined."})

if __name__ == "__main__":
    # Test printing a rule
    print(f"Rule for 'age': {get_rule('age')}")
    print(f"Rule for 'education': {get_rule('education')}")
