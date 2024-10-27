import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from tqdm import tqdm



def generate_pseudo_answer(data):
    # prompt1 = "Give the exact or most likely answer to the question below. Question: "
    # prompt2 = "Please answer the following questions in concise terms. If you're not sure of the answer, try your best to guess. Question: "
    # prompt3 = "Provide the most likely answer to the question along with your reasoning, must keep it concise. Question: "
    prompt_wow = "Based on the user's question provided below, generate a concise and informative response that includes relevant background information and maintains a friendly," \
                 " conversational tone. Here is the question:"
    question = data['question']
    response = requests.post(
        'http://10.0.102.253:8084/completion',
        json={
            'prompt': f"<|start_header_id|>user<|end_header_id|>\n\n{prompt_wow}{question}?\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            'temperature': 0
        })
    if response.status_code == 200:
        pseudo_answer = response.json().get('content', 'No answer')
        # print(pseudo_answer)
    else:
        pseudo_answer = ''
        print(f"Error: {response.status_code} - {response.text}")
    data['pseudo_answer'] = pseudo_answer
    return data


def process_dataset_concurrently(dataset):
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(generate_pseudo_answer, data) for data in dataset]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            results.append(future.result())
    return results


def main():
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    updated_dataset = process_dataset_concurrently(dataset)

    # Save the updated data set
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(updated_dataset, f, ensure_ascii=False, indent=4)
    print("Pseudo-responses are generated and added to the data set.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data i/o
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=32, help="Maximum number of parallel requests")

    args = parser.parse_args()

    main()
