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
import base64
import json
import logging
import re
from google import genai
from google.genai import types
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def strip_markdown(text):
    """Strips markdown code blocks if present."""
    match = re.search(r'```(?:[a-zA-Z]+)?\n(.*?)\n```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

client = genai.Client(vertexai=True, project=config.PROJECT_ID, location=config.LOCATION)

def generate_v0_judge():
    logging.info("Generating v0-judge.txt...")
    ref_judge = read_file("eval_prompts/reference/v0-judge.txt")
    good_img_b64 = encode_image("samples/good/img-0.png")
    poor_img_b64 = encode_image("samples/poor/img-0.png")
    
    with open("inputs/sample_inputs.json", "r") as f:
        sample_inputs = json.load(f)
    user_input = sample_inputs.get("img-0.png", "")

    meta_prompt = f"""
You are an expert Prompt Engineer. Your task is to create an EVALUATION PROMPT (Judge) for "Isometric Pixel Art Infographics".
I am providing a reference evaluation prompt used for a different task (Modern Infographics) and two sample images (one "Good" and one "Poor") for the current task.

### REFERENCE PROMPT (Modern Infographics)
{ref_judge}

### CURRENT TASK DATA
- **User Input (Prompt for Image Gen):** {user_input}
- **Good Sample Image:** (Attached)
- **Poor Sample Image:** (Attached)

### YOUR OBJECTIVE
Generate a new `v0-judge.txt` prompt specifically tailored for evaluating "Isometric Pixel Art Infographics".
1. **Maintain the Structure**: Keep the C1-C5 pillars (Layout, Art Style, Typography, Summarization, Data Viz).
2. **Output Format**: The prompt must instruct the LLM to output ONLY a JSON object.
3. **Scoring**: Each pillar (C1-C5) must have a score from 0 to 5. 
   - Use keys: `C1_score`, `C2_score`, `C3_score`, `C4_score`, `C5_score`.
   - Do NOT include an `overall_score` in the LLM output; it will be calculated externally.
4. **Tune the Rules**: Adjust the Zero-Tolerance triggers and Deduction Rubric based on what makes the "Good" sample better than the "Poor" sample in the context of pixel art infographics. 
   - Note: Unlike the reference, 3D/Isometric is DESIRED here, but it must be clean and consistent.
   - Note: Pixel art uniformity and "chunkiness" control is key.
   - Note: Scannability and Information Hierarchy are still top priorities.
5. **Strict Scoring**: Ensure the scoring remains ruthless to maximize the gap.

Output ONLY the complete prompt text for `v0-judge.txt`. Do not include any conversational filler.
"""
    
    response = client.models.generate_content(
        model=config.OPTIMIZATION_MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part(text=meta_prompt),
                    types.Part(inline_data=types.Blob(mime_type="image/png", data=good_img_b64)),
                    types.Part(inline_data=types.Blob(mime_type="image/png", data=poor_img_b64))
                ]
            )
        ]
    )
    
    content = strip_markdown(response.text)
    write_file("eval_prompts/v0-judge.txt", content)
    logging.info("v0-judge.txt created.")

def generate_critic():
    logging.info("Generating critic_contrastive.txt...")
    ref_critic = read_file("eval_prompts/reference/critic_reference.txt")
    
    meta_prompt = f"""
You are an expert Prompt Engineer. Your task is to update a CRITIC PROMPT for "Isometric Pixel Art Infographics".
The critic prompt's job is to analyze why a "Good" sample scored better than a "Poor" sample and suggest optimization rules.
Note: The overall_score is calculated as the average of five category scores (C1-C5).

### REFERENCE CRITIC PROMPT
{ref_critic}

### YOUR OBJECTIVE
Update this prompt to be specific to "Isometric Pixel Art Infographics". 
Adjust the "CORE CRITERIA" section to match the pixel art task.
Ensure the critic analyzes the breakdown of C1-C5 scores to find specific weaknesses.

Output ONLY the complete prompt text for `critic_contrastive.txt`. Do not include any conversational filler.
"""
    
    response = client.models.generate_content(
        model=config.OPTIMIZATION_MODEL,
        contents=meta_prompt
    )
    
    content = strip_markdown(response.text)
    write_file("eval_prompts/critic_contrastive.txt", content)
    logging.info("critic_contrastive.txt created.")

def generate_optimizer():
    logging.info("Generating optimizer_contrastive.txt...")
    ref_optimizer = read_file("eval_prompts/reference/optimizer_reference.txt")
    
    meta_prompt = f"""
You are an expert Prompt Engineer. Your task is to update an OPTIMIZER PROMPT for "Isometric Pixel Art Infographics".
The optimizer prompt's job is to take the critic's feedback and refine the Judge prompt.
Note: The Judge prompt we are optimizing uses the average of five category scores (C1-C5) as the final metric.

### REFERENCE OPTIMIZER PROMPT
{ref_optimizer}

### YOUR OBJECTIVE
Update this prompt to be specific to "Isometric Pixel Art Infographics". 
Adjust the "CORE DESIGN FRAMEWORK" section to match the pixel art task.
The goal is to maximize the difference in the average score between Good and Poor samples by making the C1-C5 criteria more discriminative.

Output ONLY the complete prompt text for `optimizer_contrastive.txt`. Do not include any conversational filler.
"""
    
    response = client.models.generate_content(
        model=config.OPTIMIZATION_MODEL,
        contents=meta_prompt
    )
    
    content = strip_markdown(response.text)
    write_file("eval_prompts/optimizer_contrastive.txt", content)
    logging.info("optimizer_contrastive.txt created.")

if __name__ == "__main__":
    # Ensure we are in the right directory
    if not os.path.exists("eval_prompts/reference"):
        logging.error("Could not find 'eval_prompts/reference'. Please run this script from the 'contrastive-eval-prompt-writer' directory.")
    else:
        generate_v0_judge()
        generate_critic()
        generate_optimizer()
