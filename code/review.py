"""
This script utilizes or is inspired by the code and concepts from the following GitHub repository:
Repository: AI-Scientist
URL: https://github.com/SakanaAI/AI-Scientist
Author: SakanaAI
"""


import os
import numpy as np
import json
from llm import (
    get_response_from_llm,
    get_batch_responses_from_llm,
    extract_json_between_markers,
)

it_reviewer_system_prompt = (
    "You are an AI reviewer in the IT department responsible for evaluating internal applications from employees. "
    "Your mission is to assess risks and merits related to IT tools and the infrastructure of the company. "
    "Be critical and cautious in your decision-making process. Your evaluations should be structured, concise, and actionable."
)

identification_prompt = """
Analyze the given application description to identify risks and clarify merits.

Args:
    application_description (str): A description of the application.
    
Returns:
    dict: A structured report containing identified risks and clarified merits.

Ensure that the risks and merits are clearly identified, described, and relevant to the application's context.
"""

def perform_identification(
    text,
    model,
    client,
    num_reflections=1,
    num_fs_examples=1,
    num_reviews_ensemble=1,
    temperature=0.75,
    msg_history=None,
    return_msg_history=False,
):
    if num_fs_examples > 0:
        fs_prompt = ""
        try:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            file_path = os.path.join(dir_path, "fewshot_examples/identification.txt")
            with open(file_path, "r", encoding="utf-8") as file:
                fs_prompt = file.read()
        except FileNotFoundError:
            print(f"The file at {file_path} does not exist.")
        except IOError as e:
            print(f"An error occurred while reading the file: {e}")

        base_prompt = identification_prompt + fs_prompt
    else:
        base_prompt = identification_prompt

    base_prompt += f"""
Here is the application you are asked to review:
```
{text}
```"""
    llm_review, msg_history = get_response_from_llm(
        base_prompt,
        model=model,
        client=client,
        system_message=it_reviewer_system_prompt,
        print_debug=False,
        msg_history=msg_history,
        temperature=temperature,
    )
    print(llm_review)
    review = json.loads(llm_review)

    if num_reflections > 1:
        for j in range(num_reflections - 1):
            # print(f"Relection: {j + 2}/{num_reflections}")
            text, msg_history = get_response_from_llm(
                reviewer_reflection_prompt,
                client=client,
                model=model,
                system_message=it_reviewer_system_prompt,
                msg_history=msg_history,
                temperature=temperature,
            )
            review = json.loads(text)
            assert review is not None, "Failed to extract JSON from LLM output"

            if "I am done" in text:
                # print(f"Review generation converged after {j + 2} iterations.")
                break

    if return_msg_history:
        return review, msg_history
    else:
        return review


reviewer_reflection_prompt = """Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the accuracy and soundness of the review you just created.
Include any other factors that you think are important in evaluating the paper.
Ensure the review is clear and concise, and the JSON is in the correct format.
Do not make things overly complicated.
In the next attempt, try and refine and improve your review.
Stick to the spirit of the original review unless there are glaring issues.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

If there is nothing to improve, simply repeat the previous JSON EXACTLY after the thought and include "I am done" at the end of the thoughts but before the JSON.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES."""







meta_reviewer_system_prompt = """You are an Area Chair at a machine learning conference.
You are in charge of meta-reviewing a paper that was reviewed by {reviewer_count} reviewers.
Your job is to aggregate the reviews into a single meta-review in the same format.
Be critical and cautious in your decision, find consensus, and respect the opinion of all the reviewers."""


def get_meta_review(model, client, temperature, reviews):
    # Write a meta-review from a set of individual reviews
    review_text = ""
    for i, r in enumerate(reviews):
        review_text += f"""
Review {i + 1}/{len(reviews)}:
```
{json.dumps(r)}
```
"""
    base_prompt = neurips_form + review_text

    llm_review, msg_history = get_response_from_llm(
        base_prompt,
        model=model,
        client=client,
        system_message=meta_reviewer_system_prompt.format(reviewer_count=len(reviews)),
        print_debug=False,
        msg_history=None,
        temperature=temperature,
    )
    meta_review = extract_json_between_markers(llm_review)
    return meta_review

