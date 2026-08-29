"""Stage 4 - attach GPT-4 regret labels to every target-model answer.

For each of the four answer stages, GPT-4 judges whether the answer expresses
regret. These `GPT-4-ground_truth_*` fields are the probe labels used in
stage 5.

Needs OPENAI_API_KEY in the environment.
"""
import openai
import json
import re
import argparse
import os
from tqdm import tqdm
openai.api_key = os.environ["OPENAI_API_KEY"]
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("REGRET_ROOT", os.path.dirname(_HERE))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model_size', default='7b', choices=['7b', '13b', '70b'],
                    help='Target model scale; selects the field prefix and output name.')
args = parser.parse_args()

target_model_name = 'llama-2-%s-' % args.model_size
eval_model_name = 'GPT-4-'

def generate_response(prompt, model="gpt-4o"):
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







def get_ground_truth(question, answer):
    prompt_qa = f"Based on this Question: {question} and LLM Answer: {answer}\n\n Tell me if LLM regrets answering the question? Just reply True or False, True is regret and False is no regret. Emphasis: you only need to reply True or False"
    ground_truth, _ = generate_response(prompt_qa, model="gpt-4")
    return ground_truth

target_model_name = 'llama-2-7b-'
pure_answer_key = target_model_name + "pure_answer"
initial_answer_key = target_model_name + "initial-answer_with_fake_evidence"
reflection_weak_key = target_model_name + "reflection_answer_weak"
reflection_strong_key = target_model_name + "reflection_answer_strong"

def get_standard_label_confident(input_file, output_file):
    with open(input_file, 'r', encoding="utf-8") as file:
        for line in file:
            updated_entry = json.loads(line)
            ID = updated_entry['ID']
            if ID >= 855:
                break
            emotion_mask = updated_entry['emotion_mask']
            question = updated_entry['question']
            fake_evidence = updated_entry['fake_evidence']
            ground_truth = updated_entry['ground_truth']
            weak_hint_true = updated_entry['weak_hint_true']
            strong_hint_true = updated_entry['strong_hint_true']
            pure_answer = updated_entry[pure_answer_key]
            initial_answer = updated_entry[initial_answer_key]
            reflection_weak = updated_entry[reflection_weak_key]
            reflection_strong = updated_entry[reflection_strong_key]

            # get label
            pure_ground_truth = get_ground_truth(question, pure_answer)
            initial_answer_ground_truth = get_ground_truth(question, initial_answer)
            reflection_weak_ground_truth = get_ground_truth(question, reflection_weak)
            reflection_strong_ground_truth = get_ground_truth(question, reflection_strong)

            print(f"ID: {ID}")
            updated_entry = {
                "ID": ID,
                "emotion_mask": emotion_mask,
                "question": question,
                target_model_name + "pure_answer": pure_answer,
                eval_model_name + "ground_truth_pure_answer": pure_ground_truth,
                "fake_evidence": fake_evidence,
                "ground_truth": ground_truth,
                target_model_name + "initial-answer_with_fake_evidence": initial_answer,
                eval_model_name + "ground_truth_initial": initial_answer_ground_truth,
                "weak_hint_true": weak_hint_true,
                target_model_name + "reflection_answer_weak": reflection_weak,
                eval_model_name + "ground_truth_weak": reflection_weak_ground_truth,
                "strong_hint_true": strong_hint_true,
                target_model_name + "reflection_answer_strong": reflection_strong,
                eval_model_name + "ground_truth_strong": reflection_strong_ground_truth
            }
            with open(output_file, "a", encoding="utf-8") as f:
                json.dump(updated_entry, f, ensure_ascii=False)
                f.write("\n")

def main():
    input_file = os.path.join(ROOT, 'datasets', 'generated_data_with_target_model_answer.json')
    output_file = os.path.join(ROOT, 'datasets', args.model_size + '_output_generated_data_with_target_model_answer_label_confident.json')
    # if os.path.exists(output_file):
    get_standard_label_confident(input_file, output_file)
    print(f"Labelled data written to {output_file}")

if __name__ == "__main__":
    main()