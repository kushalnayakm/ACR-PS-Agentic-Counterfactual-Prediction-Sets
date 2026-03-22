import json
import os
from dotenv import load_dotenv

load_dotenv()

class FAXAgent:
    def __init__(self, model_name='gemini-2.0-flash'):
        self.nv_key = os.getenv("NVIDIA_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        
        if self.nv_key:
            print("FAX Agent initialized with NVIDIA API.")
        elif self.gemini_key:
            print(f"FAX Agent initialized with Gemini API (Model: {self.model_name}).")
        else:
            print("Warning: No API keys found. Using local fallback.")

    def load_filtered_data(self, file_path="acr/filtered_counterfactuals.json"):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Filtered data not found at {file_path}")
        with open(file_path, 'r') as f:
            return json.load(f)

    def generate_narrative(self, sample_data):
        """Generate a plain English counterfactual explanation"""
        original = sample_data['original_data']
        valid_cfs = sample_data['valid_counterfactuals']
        invalid = sample_data['invalid_suggestions']
        
        context = f"Current Profile: {json.dumps(original, indent=2)}\n\n"
        
        if invalid:
            context += "The following improvements were considered but discarded as impossible:\n"
            for inv in invalid:
                context += f"- {inv['reason']}\n"
        
        context += "\nActionable suggestions leading to a positive outcome:\n"
        for cf in valid_cfs:
            changes = []
            for feat, val in cf.items():
                if val != original.get(feat):
                    changes.append(f"{feat}: {original.get(feat)} -> {val}")
            context += f"- {', '.join(changes)}\n"

        prompt = f"""You are a helpful financial advisor (FAX Module). You provide faithful, actionable counterfactual explanations. Never suggest changing immutable features like age, race, or gender.

Based on the data below, explain to the user why they were rejected (income <= 50K) and what realistic steps they can take to be approved (income > 50K).

{context}

Draft a friendly, 3-4 sentence explanation for the user."""

        # 1. Try NVIDIA
        if self.nv_key:
            try:
                from openai import OpenAI
                client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=self.nv_key)
                response = client.chat.completions.create(
                    model="meta/llama-3.1-70b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7, max_tokens=256
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"  NVIDIA Error: {e}")

        # 2. Try Gemini
        if self.gemini_key:
            try:
                from google import genai
                client = genai.Client(api_key=self.gemini_key)
                response = client.models.generate_content(model=self.model_name, contents=prompt)
                return response.text
            except Exception as e:
                print(f"  Gemini Error: {e}")

        # 3. Local Fallback
        return self._generate_local_narrative(original, valid_cfs, invalid)

    def _generate_local_narrative(self, original, valid_cfs, invalid):
        """Generate narrative locally without LLM (fallback)"""
        narrative = "Based on your current profile, here's what we found:\n"
        
        if invalid:
            narrative += "\nWe considered some changes but they aren't realistic:\n"
            for inv in invalid:
                narrative += f"  - {inv['reason']}\n"
        
        if valid_cfs:
            narrative += "\nHere are actionable steps you CAN take:\n"
            for cf in valid_cfs:
                changes = []
                for feat, val in cf.items():
                    if val != original.get(feat):
                        changes.append(f"Change {feat} from '{original.get(feat)}' to '{val}'")
                if changes:
                    narrative += f"  - {'; '.join(changes)}\n"
        
        narrative += "\nThese suggestions respect your immutable characteristics (age, race, gender) and focus only on factors you can realistically change."
        return narrative

    def run_all(self):
        data = self.load_filtered_data()
        all_narratives = []
        
        for sample in data:
            print(f"\nGenerating narrative for Sample {sample['sample_id']}...")
            narrative = self.generate_narrative(sample)
            print("-" * 50)
            print(narrative)
            print("-" * 50)
            all_narratives.append({
                "sample_id": sample['sample_id'],
                "narrative": narrative
            })
            
        with open("acr/final_narratives.json", 'w') as f:
            json.dump(all_narratives, f, indent=4)
        
        print(f"\nAll narratives saved to acr/final_narratives.json")

if __name__ == "__main__":
    agent = FAXAgent() 
    agent.run_all()
