"""
LLM Narrator - Generates plain-English explanations using Gemini.
Integrated into the ACR Dashboard.
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()


def get_narrative(query_dict, valid_cfs, invalid_cfs, feature_names):
    """
    Narrate the audit results using NVIDIA (Llama 3.1) or Gemini.
    Falls back to a local template if APIs are unavailable.
    """
    # 1. Build context
    context = f"Person's Profile: {json.dumps(query_dict, default=str)}\n\n"

    if invalid_cfs:
        context += "REJECTED suggestions (impossible/unfaithful):\n"
        for item in invalid_cfs:
            context += f"- {item['reason']}\n"

    if valid_cfs:
        context += "\nAPPROVED suggestions (actionable):\n"
        for cf in valid_cfs:
            changes = []
            for f in feature_names:
                if f in cf and str(cf[f]) != str(query_dict.get(f)):
                    changes.append(f"{f}: {query_dict.get(f)} → {cf[f]}")
            if changes:
                context += f"- {', '.join(changes)}\n"

    prompt = f"""You are a friendly, professional advisor explaining AI-generated suggestions to a user.

CONTEXT:
{context}

RULES:
- Never suggest changing immutable features (age, race, sex, genetics)
- Explain WHY certain suggestions were rejected (they violated causal rules)
- Highlight the actionable steps the user CAN take
- Be encouraging and constructive
- Keep it to 4-5 sentences maximum

Write a clear, helpful explanation for this person:"""

    # 2. Try NVIDIA (Llama 3.1)
    nv_key = os.getenv("NVIDIA_API_KEY")
    if nv_key:
        try:
            from openai import OpenAI
            client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=nv_key)
            response = client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7, max_tokens=256
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"NVIDIA Error: {e}")

    # 3. Try Gemini (Fallback)
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
            return response.text
        except Exception as e:
            print(f"Gemini Error: {e}")

    # 4. Local Fallback (Final)
    return _local_narrative(query_dict, valid_cfs, invalid_cfs, feature_names)


def _local_narrative(query_dict, valid_cfs, invalid_cfs, feature_names):
    """Fallback: generate a structured narrative without LLM."""
    parts = ["Based on the causal analysis of your profile:\n"]

    if invalid_cfs:
        parts.append(f"**{len(invalid_cfs)} suggestion(s) were discarded** because they tried to change things that cannot be changed (like age or genetics).\n")

    if valid_cfs:
        parts.append(f"**{len(valid_cfs)} actionable suggestion(s) remain.** Here's what you can realistically do:\n")
        for cf in valid_cfs:
            changes = []
            for f in feature_names:
                if f in cf and str(cf[f]) != str(query_dict.get(f)):
                    changes.append(f"adjust **{f}** from {query_dict.get(f)} to {cf[f]}")
            if changes:
                parts.append(f"- {', '.join(changes)}\n")
    else:
        parts.append("Unfortunately, no actionable suggestions could pass the causal audit. All generated options required changing immutable features.\n")

    parts.append("\n*These suggestions respect your immutable characteristics and focus only on factors within your control.*")
    return ''.join(parts)
