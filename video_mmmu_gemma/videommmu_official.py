"""
Official Video-MMMU prompts and scoring, from lmms-eval's video_mmmu task
(lmms_eval/tasks/videommmu/{_default_template_yaml,utils.py}), https://github.com/EvolvingLMMs-Lab/lmms-eval

- Prompts, open-ended parsing and judging: identical at the original task commit
  74b6d95ce7ca83c5ed0883581d91a19fcb5b66d0 (Feb 2025) and at 08affba133b97b19351897f4d42081ead2b3412e (Oct 2026).
- Multiple-choice parser: the ORIGINAL Video-MMMU parser from 74b6d95ce7 (used for the published leaderboard,
  later moved unchanged to mmmu_mcq_utils.parse_videommmu_multi_choice_response). lmms-eval replaced it in
  April 2026 (9ca4445d0c) with a strict shared extractor that abstains on answers like "**B. 3, 4**", which is
  how Gemma states its choice.

Only the text/scoring logic is copied; the lmms-eval video-path lookup is not used.
"""
import re

import numpy as np

# ---- _default_template_yaml: lmms_eval_specific_kwargs.default ----
PRE_PROMPT = "You should watch and learn the video content. Then apply what you learned to "
PERCEPTION_AND_COMPREHENSION_PROMPT = "\nPlease ignore the Quiz question in last frame of the video."
MCQ_PROMPT = "answer the following multi-choice question. The image for this question is at the end of the video.\n"
OPEN_ENDED_PROMPT = "answer the following open-ended question. The image for this question is at the end of the video.\n"
MCQ_POST_PROMPT = ""
MAX_NEW_TOKENS = 1024  # generation_kwargs.max_new_tokens


# ---- utils.py: prompt construction ----
def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    if all(option.startswith(f"{letter}.") for option, letter in zip(options, option_letters)):
        return "\n".join(options)
    return "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])


def doc_to_text_adaptation(doc):
    pre_prompt = PRE_PROMPT
    question = doc["question"]
    if doc["question_type"] == "multiple-choice":
        pre_prompt += MCQ_PROMPT
        question += "\n" + parse_options(doc["options"])
        return f"{pre_prompt}{question}{MCQ_POST_PROMPT}"
    pre_prompt += OPEN_ENDED_PROMPT
    return f"{pre_prompt}{question}"


def doc_to_text_perception_comprehension(doc):
    question = doc["question"] + "\n" + parse_options(doc["options"])
    return f"{question}{PERCEPTION_AND_COMPREHENSION_PROMPT}{MCQ_POST_PROMPT}"


def doc_to_text(doc, track):
    return doc_to_text_adaptation(doc) if track == "Adaptation" else doc_to_text_perception_comprehension(doc)


# ---- utils.py: answer parsing ----
def get_multi_choice_info(options, start_chr="A"):
    all_choices, index2ans = [], {}
    for i, option in enumerate(options):
        choice = chr(ord(start_chr) + i)
        index2ans[choice] = option
        all_choices.append(choice)
    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    if response == "API Error" or response == "":
        return "API Error"

    # Step 1: Clean up punctuation from the response
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # Add space to avoid partial match
    # print(response)

    index_ans = True
    ans_with_brack = False
    ans_with_period = False
    ans_with_colon = False
    candidates = []

    # Step 2: If no candidates, look for choices with a period after (A. B. C. D.)
    for choice in all_choices:  # e.g., A. B. C. D.
        if f"{choice}." in response:
            candidates.append(choice)
            ans_with_period = True
    # Step 2.1: If no candidates, look for choices with a colon after (A: B: C: D:)
    for choice in all_choices:  # e.g., A: B: C: D:
        if f"{choice}:" in response:
            candidates.append(choice)
            ans_with_colon = True
    # Step 3: Look for choices with parentheses e.g., (A) (B) (C) (D)
    if len(candidates) == 0:
        for choice in all_choices:  # e.g., (A) (B) (C) (D)
            if f"({choice})" in response:
                candidates.append(choice)
                ans_with_brack = True
    # Step 4: If no candidates, look for choices with a space after (A B C D)
    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    # Step 5: If no candidates and response has more than 5 tokens, try parsing based on content
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # It's content answer, not an index

    # Step 6: If still no candidates, randomly choose one
    if len(candidates) == 0:
        pred_index = "No Answer Found."

    # Step 7: If multiple candidates found, use the one appearing last
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_period:
                for can in candidates:
                    index = response.rfind(f"{can}.")
                    start_indexes.append(index)
            elif ans_with_colon:
                for can in candidates:
                    index = response.rfind(f"{can}:")
                    start_indexes.append(index)
            elif ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # Get the last one (max index)
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        # If only one candidate, use it
        pred_index = candidates[0]

    return pred_index


def extract_numbers(string):
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"
    numbers_with_commas = re.findall(pattern_commas, string)
    numbers_scientific = re.findall(pattern_scientific, string)
    numbers_simple = re.findall(pattern_simple, string)
    return numbers_with_commas + numbers_scientific + numbers_simple


def check_is_number(string):
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        return False


def normalize_str(string):
    string = string.strip()
    if check_is_number(string):
        string = string.replace(",", "")
        string = float(string)
        string = round(string, 2)
        return [string]
    string = string.lower()
    if len(string) == 1:
        return [" " + string, string + " "]
    return [string]


def parse_open_response(response):
    if response == "API Error" or response == "":
        return "API Error"

    def get_key_subresponses(response):
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", response)
        indicators_of_keys = [
            "could be ", "so ", "is ", "thus ", "therefore ", "final ", "answer ", "result ", "are ",
            "in total ", "total ", "identify ", "recognize ", "calculated as ", "counted as ", "measured as ",
            "observed as ", "concluded as ", "found to be ", "equals ", "determined to be ", "number of ",
            "value is ", "adds up to ", "have ", "has ",
        ]
        key_responses = []
        for index, resp in enumerate(sub_responses):
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(["="])
            shortest_key_response = None
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    elif len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                        shortest_key_response = resp.split(indicator)[-1].strip()
            if shortest_key_response:
                if shortest_key_response.strip() not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:
            return [response]
        return key_responses

    key_responses = get_key_subresponses(response)
    pred_list = key_responses.copy()
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))
    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))
    return list(set(tmp_pred_list))


# ---- utils.py: judging ----
def eval_multi_choice(gold_i, pred_i):
    if isinstance(gold_i, list):
        return any(answer == pred_i for answer in gold_i)
    return gold_i == pred_i


def eval_open(gold_i, pred_i):
    if isinstance(gold_i, list):
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)
    for pred in pred_i:
        if isinstance(pred, str):
            for norm_ans in norm_answers:
                if isinstance(norm_ans, str) and norm_ans in pred:
                    return True
        else:
            if pred in norm_answers:
                return True
    return False


def strip_special_tokens(response):
    """Drop chat-template markers (e.g. Gemma's trailing "<turn|>") that decoding can leave in the text."""
    return re.sub(r"<\|?turn\|?>|<eos>|<end_of_turn>", "", str(response)).strip()


def score(doc, response):
    """videommmu_process_results + evaluate_videommmu for one doc -> (parsed_pred, correct)."""
    response = strip_special_tokens(response)
    if doc["question_type"] == "multiple-choice":
        index2ans, all_choices = get_multi_choice_info(doc["options"])
        parsed = parse_multi_choice_response(response, all_choices, index2ans)
        return parsed, eval_multi_choice(doc["answer"], parsed)
    parsed = parse_open_response(response)
    if parsed == "API Error":  # empty output; upstream drops these from aggregation, we count them wrong
        return parsed, False
    return parsed, eval_open(doc["answer"], parsed)


def extract_subset_name(input_string):
    split = input_string.split("_")[0]
    match = re.compile(rf"^{split}_(.+?)_\d+$").search(input_string)
    if match:
        return match.group(1)
    raise ValueError(f'No match found in "{input_string}"')


DOMAIN_CAT2SUB_CAT = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health and Medicine": ["Basic_Medical_Science", "Clinical_Medicine", "Diagnostics_and_Laboratory_Medicine", "Pharmacy", "Public_Health"],
    "Humanities and Social Science": ["History", "Literature", "Sociology", "Psychology"],
    "Tech and Engineering": ["Agriculture", "Architecture_and_Engineering", "Computer_Science", "Electronics", "Energy_and_Power", "Materials", "Mechanical_Engineering"],
}
SUB_CAT2DOMAIN = {sub: dom for dom, subs in DOMAIN_CAT2SUB_CAT.items() for sub in subs}
