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


# ### 4_check_status.py ###

# def align_prompts(new_version):
#     """Aligns critic and optimizer prompts with the new judge prompt."""
#     setup_logging()
#     client = genai.Client(vertexai=True, project=config.PROJECT_ID, location=config.LOCATION)
    
#     judge_prompt_path = f"eval_prompts/{new_version}-judge.txt"
#     judge_content = read_file(judge_prompt_path)
    
#     if not judge_content:
#         logging.error(f"New judge prompt not found at {judge_prompt_path}")
#         return

#     # 1. Update Critic Prompt
#     logging.info(f"Aligning critic prompt with {new_version}-judge...")
#     ref_critic = read_file("eval_prompts/reference/critic_contrastive.txt")
    
#     critic_align_prompt = f"""
# You are an expert Prompt Engineer. I have a new version of an EVALUATION PROMPT (Judge). 
# You need to update the CRITIC PROMPT template so it remains aligned with the Judge's criteria.

# ### NEW JUDGE PROMPT
# {judge_content}

# ### EXISTING CRITIC TEMPLATE
# {ref_critic}

# ### YOUR TASK
# Update the "CORE CRITERIA" section of the CRITIC TEMPLATE to match the pillars and logic of the NEW JUDGE PROMPT. 
# Ensure that placeholders like &USER_INPUT&, &GOOD_EVAL&, &POOR_EVAL&, &GOOD_SCORE&, &POOR_SCORE&, and &SCORE_DIFF& are preserved.
# Maintain the JSON output format.

# Output ONLY the complete updated CRITIC PROMPT text.
# """
    
#     response = client.models.generate_content(model=config.OPTIMIZATION_MODEL, contents=critic_align_prompt)
#     new_critic_content = strip_markdown(response.text)
#     write_file(f"eval_prompts/{new_version}-critic_contrastive.txt", new_critic_content)
#     logging.info(f"{new_version}-critic_contrastive.txt created and aligned.")

#     # 2. Update Optimizer Prompt
#     logging.info(f"Aligning optimizer prompt with {new_version}-judge...")
#     ref_optimizer = read_file("eval_prompts/reference/meta_eval_optimizer.txt")
    
#     optimizer_align_prompt = f"""
# You are an expert Prompt Engineer. I have a new version of an EVALUATION PROMPT (Judge). 
# You need to update the OPTIMIZER PROMPT template so it remains aligned with the Judge's criteria.

# ### NEW JUDGE PROMPT
# {judge_content}

# ### EXISTING OPTIMIZER TEMPLATE
# {ref_optimizer}

# ### YOUR TASK
# Update the "CORE DESIGN FRAMEWORK" section of the OPTIMIZER TEMPLATE to match the pillars of the NEW JUDGE PROMPT.
# Ensure that the {{structured_context}} placeholder is preserved.

# Output ONLY the complete updated OPTIMIZER PROMPT text.
# """
    
#     response = client.models.generate_content(model=config.OPTIMIZATION_MODEL, contents=optimizer_align_prompt)
#     new_optimizer_content = strip_markdown(response.text)
#     write_file(f"eval_prompts/{new_version}-meta_eval_optimizer.txt", new_optimizer_content)
#     logging.info(f"{new_version}-meta_eval_optimizer.txt created and aligned.")
