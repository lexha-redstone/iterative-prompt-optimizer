# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import argparse
import re
from google import genai
from google.genai import types
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_file(path):
    if not os.path.exists(path):
        logging.warning(f"File not found: {path}")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def strip_markdown(text):
    """Strips markdown code blocks if present."""
    match = re.search(r'```(?:[a-zA-Z]+)?\n(.*?)\n```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def fix_placeholders(text):
    """Ensures consistent placeholders across prompts."""
    # Replace &PROMPT& or [USER_INPUT] with &USER_INPUT&
    text = text.replace("&PROMPT&", "&USER_INPUT&")
    text = text.replace("[USER_INPUT]", "&USER_INPUT&")
    return text

client = genai.Client(vertexai=True, project=config.PROJECT_ID, location=config.LOCATION)

def generate_inference_v0():
    logging.info(f"Generating inference_v0.txt...")
    
    # Get judge path from config.py
    judge_path = getattr(config, 'EVAL_PROMPT', "prompts/v7-judge.txt")
    if not os.path.exists(judge_path):
        logging.warning(f"Judge path missing: {judge_path}. Falling back to default v7-judge.")
        judge_path = "prompts/v7-judge.txt"
    
    current_judge = read_file(judge_path)
    
    # Get 1-2 images from GOLDEN_STANDARD_DIR
    golden_dir = config.GOLDEN_STANDARD_DIR
    image_parts = []
    if os.path.exists(golden_dir):
        try:
            images = [f for f in os.listdir(golden_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_name in images[:2]:
                img_path = os.path.join(golden_dir, img_name)
                with open(img_path, "rb") as f:
                    img_data = f.read()
                mime_type = "image/png" if img_name.lower().endswith('.png') else "image/jpeg"
                image_parts.append(types.Part.from_bytes(data=img_data, mime_type=mime_type))
                logging.info(f"Included golden standard image: {img_name}")
        except Exception as e:
            logging.warning(f"Error reading golden standard images: {e}")
    else:
        logging.warning(f"Golden standard directory not found: {golden_dir}")

    meta_prompt = f"""
You are an expert Prompt Engineer for Image Generation models.
Your task is to create an initial version (v0) of an "Inference Prompt" (Create Prompt) for "Infographics".

This inference prompt will take a user's concept via the `&USER_INPUT&` placeholder and generate a high-quality infographic.

### EVALUATION CRITERIA (The Judge)
The generated image will be evaluated against these strict rules:
{current_judge}

### REFERENCE STYLE (Golden Standards)
Attached are 1-2 images that represent the "Golden Standard" for this style. 
Your generated prompt must guide the model to produce images with similar aesthetic qualities, layout principles, and technical execution.

### YOUR OBJECTIVE
Generate a comprehensive inference prompt.
1. **Placeholder**: Use `&USER_INPUT&` exactly as the placeholder for the input concept.
2. **Detailed Style Instructions**: Based on the Judge's criteria and the Golden Standard images, provide specific instructions on layout, typography, color palettes, and visual elements.
3. **Negative Prompting**: Include instructions on what to avoid to prevent common failures identified in the evaluation criteria.

Output ONLY the complete prompt text. Do not include any conversational filler or markdown blocks.
"""

    response = client.models.generate_content(
        model=config.OPTIMIZATION_MODEL,
        contents=[meta_prompt] + image_parts
    )
    
    content = fix_placeholders(strip_markdown(response.text))
    
    target_path = f"prompts/inference_v0.txt"
    write_file(target_path, content)
    logging.info(f"{target_path} created.")

def generate_critic():
    logging.info(f"Generating critic.txt...")
    ref_critic = read_file("prompts/reference/critic_reference.txt")
    
    # Get judge path from config.py
    judge_path = getattr(config, 'EVAL_PROMPT', "prompts/v7-judge.txt")
    if not os.path.exists(judge_path):
        logging.warning(f"Judge path missing: {judge_path}. Falling back to default v7-judge.")
        judge_path = "prompts/v7-judge.txt"
    
    current_judge = read_file(judge_path)
    
    if not ref_critic or not current_judge:
        logging.error(f"Missing necessary reference or current judge prompt. Cannot generate critic.")
        return

    meta_prompt = f"""
You are an expert Prompt Engineer. Your task is to create a CRITIC PROMPT for "Infographics".
The critic prompt's job is to analyze why a "Generated Image" failed compared to the "User Input" and the "Art Direction" defined in the evaluation criteria.

### REFERENCE CRITIC PROMPT
{ref_critic}

### CURRENT EVALUATION PROMPT (Judge Prompt - {os.path.basename(judge_path)})
{current_judge}

### YOUR OBJECTIVE
Generate a new `critic.txt` prompt tailored for the "Infographics" task.
1. **Update Core Constraints**: Replace the "CORE CONSTRAINTS" section in the reference with the specific visual rules and art direction found in the Current Evaluation prompt. Focus on the strict grid, flatness, color codes, and typography rules (e.g., League Gothic, All-Caps headers, flush-left).
2. **Incorporate Judge's Ruthlessness**: Notice if the Judge uses "Ruthless Bifurcated Scoring" or "AUTO 0.0" triggers. The Critic must be equally analytical and offensive in identifying these exact failure patterns.
3. **Vision Gap Analysis**: Explicitly instruct the Critic to compare the "TARGET" image against the "GOLDEN STANDARD" image. It must identify the "Vision Gap"—the specific aesthetic or structural distance between the failure and perfection.
4. **Maintain the Structure**: Keep the &USER_INPUT&, [EVAL_RESULT], and the expanded Output JSON format.

**OUTPUT FORMAT:**
Respond ONLY with a valid JSON object matching the exact structure below.

{{
  "analysis": "A concise breakdown of where the generation succeeded or failed.",
  "vision_gap_analysis": "A detailed comparison between the Golden Standard and the Target image, identifying the specific aesthetic distance and missing quality.",
  "good_dna_to_preserve": [
    "Element 1 that worked well and should not be changed",
    "Element 2"
  ],
  "prompt_vulnerabilities": [
    "Specific weakness 1 (e.g., 'Generator failed typographic scale constraints...')",
    "Specific weakness 2"
  ],
  "prompt_optimization_rules": [
    "Rule 1: Direct, explicit instruction to add to the inference prompt to patch the vulnerability.",
    "Rule 2"
  ]
}}

Output ONLY the complete prompt text. Do not include any conversational filler or markdown blocks.
"""
    
    response = client.models.generate_content(
        model=config.OPTIMIZATION_MODEL,
        contents=meta_prompt
    )
    
    content = fix_placeholders(strip_markdown(response.text))
    
    # Save as the base critic file
    target_path = getattr(config, 'CRITIC_PROMPT', "prompts/critic.txt")
    if isinstance(target_path, dict):
        target_path = "prompts/critic.txt"
    write_file(target_path, content)
    logging.info(f"{target_path} created using judge {judge_path}.")

def generate_optimizer():
    logging.info(f"Generating meta_prompt.txt...")
    ref_optimizer = read_file("prompts/reference/optimizer_reference.txt")
    
    # Get judge path from config.py
    judge_path = getattr(config, 'EVAL_PROMPT', "prompts/v7-judge.txt")
    if not os.path.exists(judge_path):
        logging.warning(f"Judge path missing: {judge_path}. Falling back to default v7-judge.")
        judge_path = "prompts/v7-judge.txt"
        
    current_judge = read_file(judge_path)
    
    if not ref_optimizer or not current_judge:
        logging.error(f"Missing necessary reference or current judge prompt. Cannot generate optimizer.")
        return

    meta_prompt = f"""
You are an expert Prompt Engineer. Your task is to create an OPTIMIZER PROMPT (Meta-Prompt) for "Infographics".
The optimizer prompt's job is to take the critic's feedback and refine the inference (Create) prompt.

### REFERENCE OPTIMIZER PROMPT
{ref_optimizer}

### CURRENT EVALUATION PROMPT (Judge Prompt - {os.path.basename(judge_path)})
{current_judge}

### YOUR OBJECTIVE
Generate a new `meta_prompt.txt` prompt tailored for the "Infographics" task.
1. **Update Context**: Adjust the instructions to reflect the specific visual constraints and "Bifurcated Scoring" logic of the current Judge.
2. **Maintain the Structure**: Keep the sections: Visual Error Pattern Analysis, Hypotheses for Improvement, and Optimized Prompt Proposal.
3. **Refine Guidance**: Ensure the optimizer knows how to balance negative and positive constraints for this specific style, especially to avoid "AUTO 0.0" triggers.
4. **Consistency**: Ensure `&USER_INPUT&` is used for the input text placeholder.

Output ONLY the complete prompt text. Do not include any conversational filler or markdown blocks.
"""
    
    response = client.models.generate_content(
        model=config.OPTIMIZATION_MODEL,
        contents=meta_prompt
    )
    
    content = fix_placeholders(strip_markdown(response.text))
    
    # Save as the base meta_prompt file
    target_path = getattr(config, 'META_PROMPT', "prompts/meta_prompt.txt")
    if isinstance(target_path, dict):
        target_path = "prompts/meta_prompt.txt"
    write_file(target_path, content)
    logging.info(f"{target_path} created using judge {judge_path}.")

if __name__ == "__main__":
    generate_inference_v0()
    generate_critic()
    generate_optimizer()
