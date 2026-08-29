"""Stage 1 - build the regret dataset with the three-stage prompting workflow.

For every ConflictQA item the script produces:
  a0  the model's answer with no evidence at all ("pure" answer),
  a1  the answer after being shown strengthened *fake* evidence,
  a2  the answer after an additional *weak hint*,
  a3  the answer after the *true* evidence (parametric memory) is revealed.

Each answer is paired with a self-reported confidence score and a GPT-4o-mini
judgement of whether the answer expresses regret. The result is the regret
dataset released in `data/regret_dataset.json`.

Usage:
    export OPENAI_API_KEY=...
    python src/01_build_regret_dataset.py \
        --input data/conflictQA-strategyQA-gpt4.json \
        --output data/regret_dataset.json

Note: the released dataset was produced with the `openai==0.28.x` client and
temperature 0.7, so regeneration will not be token-identical.
"""

import argparse
import json
import os
import re

import openai
from tqdm import tqdm

openai.api_key = os.environ["OPENAI_API_KEY"]

# Models used for each role in the workflow (see paper, Section 3.1).
EVIDENCE_MODEL = "gpt-4o"       # writes and strengthens the fake evidence
HINT_MODEL = "gpt-4"            # writes the weak hint
ANSWER_MODEL = "gpt-4o-mini"    # answers the questions and judges regret


def read_data_from_json(file_path):
    """Read a JSON-lines file into a list of dicts."""
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line))
    return data_list


def generate_response(prompt, model="gpt-4o"):
    """Return (answer, self-reported confidence score in 1..10).

    The confidence is obtained with a second call that asks the model to rate
    its own answer; if no number can be parsed the score defaults to 5.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Question: {prompt}"}
            ],
            max_tokens=500,
            n=1,
            temperature=0.7,
        )
        answer = response.choices[0].message['content'].strip()

        confidence_prompt = f"On a scale of 1 to 10, how confident are you in the following answer? Provide only a number. Answer: {answer}"
        confidence_response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": confidence_prompt}
            ],
            max_tokens=2,
            n=1,
            temperature=0.7,
        )
        confidence_text = confidence_response.choices[0].message['content'].strip()

        match = re.search(r'\d+', confidence_text)
        confidence_score = int(match.group()) if match else 5
        return answer, confidence_score
    except Exception as e:
        print(f"Error while generating a response: {e}")
        return None, None


def generate_fake_evidence(question, ground_truth):
    """Write plausible-sounding evidence for the opposite of the ground truth."""
    prompt = f"""
    Generate fake evidence that supports the opposite of the following ground truth: {ground_truth}
    The evidence should be related to the question: {question}
    Ensure the evidence is false and misleading, but still sounds plausible.
    Provide only the fake evidence, without any additional explanations or introductions.
    """
    fake_evidence, _ = generate_response(prompt, model=EVIDENCE_MODEL)
    return fake_evidence


def strengthen_fake_evidence(fake_evidence):
    """Make the fake evidence more convincing while keeping it false."""
    prompt = f"""
    The following is fake evidence: {fake_evidence}
    Strengthen this fake evidence to make it more convincing, but ensure it remains false and misleading.
    Provide only the strengthened fake evidence, without any additional explanations or introductions.
    """
    strengthened_fake_evidence, _ = generate_response(prompt, model=EVIDENCE_MODEL)
    return strengthened_fake_evidence


def generate_weak_hint(question, ground_truth, fake_evidence, strong_hint_true):
    """Write an indirect hint that nudges the model to reconsider a1."""
    prompt = f"""
    The question is: {question}
    The correct answer is: {ground_truth}
    The following is fake evidence: {fake_evidence}
    The following is true evidence: {strong_hint_true}
    
    Your task is to generate a weak hint that subtly encourages the model to reflect on the fake evidence.
    The hint should meet the following criteria:
    1. It should not directly reveal the correct answer or the true evidence.
    2. It should not explicitly contradict the fake evidence.
    3. It should provide an indirect or metaphorical clue that might lead the model to question the fake evidence.
    4. It should be neutral and open-ended, encouraging broader thinking.

    Provide only the weak hint, without any additional explanations or introductions.
    """
    weak_hint, _ = generate_response(prompt, model=HINT_MODEL)
    return weak_hint


def get_ground_truth(question, answer):
    """Ask the judge model whether `answer` expresses regret ("True"/"False")."""
    prompt_qa = f"Based on this Question: {question} and LLM Answer: {answer}\n\n Tell me if LLM regrets answering the question? Just reply True or False, True is regret and False is no regret. Emphasis: you only need to reply True or False"
    ground_truth, _ = generate_response(prompt_qa, model=ANSWER_MODEL)
    return ground_truth


def generate_data(input_file, output_file):
    """Run the three-stage workflow over every item and append JSON lines."""
    data_list = read_data_from_json(input_file)
    for data in tqdm(data_list, desc="Generating Data", unit="item"):
        question = data["question"]
        ground_truth = data["ground_truth"][0]
        strong_hint_true = data["parametric_memory"]

        # a0: answer with no evidence supplied.
        original_prompt = f"Question: {question}\nProvide a direct answer in 1-5 sentences, focusing only on answering the specific question."
        Pure_answer, confidence_score_pure = generate_response(original_prompt, model=ANSWER_MODEL)
        pure_ground_truth = get_ground_truth(question, Pure_answer)

        # Fake evidence, then strengthened.
        fake_evidence = generate_fake_evidence(question, ground_truth)
        strengthened_fake_evidence = strengthen_fake_evidence(fake_evidence)

        # a1: answer conditioned on the fake evidence (induces misinformation).
        initial_prompt = f"Based on this context: {strengthened_fake_evidence}\n\nQuestion: {question}\nProvide a direct answer in 1-5 sentences, focusing only on answering the specific question."
        initial_answer, confidence_score_initial = generate_response(initial_prompt, model=ANSWER_MODEL)
        fake_evidence_initial_ground_truth = get_ground_truth(question, initial_answer)
        if initial_answer is None:
            continue

        weak_hint_true = generate_weak_hint(question, ground_truth, fake_evidence, strong_hint_true)

        # a2: answer after the weak hint.
        weak_prompt = f"""
        You previously answered the following question: {question}
        Your initial answer was: {initial_answer}
        Now, consider this hint: {weak_hint_true}
        Do you regret your previous answer? Provide a direct answer in 1-5 sentences, focusing only on answering the specific question.
        """
        reflection_answer_weak, confidence_score_weak = generate_response(weak_prompt, model=ANSWER_MODEL)
        weak_hint_res_ground_truth = get_ground_truth(question, reflection_answer_weak)

        # a3: answer after the true evidence is revealed.
        strong_prompt = f"""
        You previously answered the following question: {question}
        Your initial answer was: {initial_answer}
        After considering the following hint: {weak_hint_true}
        Your revised answer was: {reflection_answer_weak}
        Now, consider this information: {strong_hint_true}
        Which previous answer do you regret?What's the final answer? Provide a direct answer in 1-5 sentences, focusing only on answering the specific question.
        """
        reflection_answer_strong, confidence_score_strong = generate_response(strong_prompt, model=ANSWER_MODEL)
        strong_hint_res_ground_truth = get_ground_truth(question, reflection_answer_strong)

        # Field names are kept byte-for-byte as they appear in the released
        # dataset, including the three keys that end in a colon and the
        # duplicated "gpt-4o-mini-confidence_score_initial" key. The duplicate
        # is intentional: Python keeps the later value, so the stored field is
        # the confidence of a1 and the confidence of the pure answer is not
        # retained. Renaming or de-duplicating either key would make this
        # script disagree with data/regret_dataset.json.
        generated_data = {
            "question": question,
            "Pure_answer": Pure_answer,
            "gpt-4o-mini-confidence_score_initial": confidence_score_pure,
            "pure_ground_truth": pure_ground_truth,
            "fake_evidence": strengthened_fake_evidence,
            "ground_truth": ground_truth,
            "gpt-4o-mini-initial_answer_with_fake_evidence": initial_answer,
            "fake_evidence_initial_ground_truth:": fake_evidence_initial_ground_truth,
            "gpt-4o-mini-confidence_score_initial": confidence_score_initial,
            "weak_hint_true": weak_hint_true,
            "gpt-4o-mini-reflection_answer_weak": reflection_answer_weak,
            "gpt-4o-mini-confidence_score_weak": confidence_score_weak,
            "weak_hint_res_ground_truth:": weak_hint_res_ground_truth,
            "strong_hint_true": strong_hint_true,
            "gpt-4o-mini-reflection_answer_strong": reflection_answer_strong,
            "gpt-4o-mini-confidence_score_strong": confidence_score_strong,
            "strong_hint_res_ground_truth:": strong_hint_res_ground_truth,
        }

        with open(output_file, "a", encoding="utf-8") as f:
            json.dump(generated_data, f, ensure_ascii=False)
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="data/conflictQA-strategyQA-gpt4.json",
                        help="ConflictQA source file (JSON lines).")
    parser.add_argument("--output", default="data/regret_dataset.json",
                        help="Destination file (JSON lines); removed if it exists.")
    args = parser.parse_args()

    if os.path.exists(args.output):
        os.remove(args.output)
    generate_data(args.input, args.output)
    print(f"Dataset written to {args.output}")


if __name__ == "__main__":
    main()
