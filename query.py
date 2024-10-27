"""Query Model on Preprocessed Data (input - output)."""

import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import datasets
from utils.util import write_dataset
from peft import PeftModel


def clean_output(text: str, prefix: str = None) -> str:
    """Extract out the answer text portion.

    prediction: e.g., "The above context is (-23.0) helpful. answer: Violet Alva"
    returns: e.g., "Violent Alva"
    """
    if (prefix is not None) and text.startswith(prefix):
        index = len(prefix)
        text = text[index:].lstrip()

    text = text.replace("Answer", "answer")
    if "<extra_id_0>" in text:
        text = text.split("<extra_id_0>")[0].strip()
    if "answer:" in text:
        text = text.split("answer:")[-1].strip()

    text = text.replace("Response", "response")
    if "response:" in text:
        text = text.split("response:")[-1].strip()

    text = text.replace("Judgement", "judgement")
    if "judgement:" in text:
        text = text.split("judgement:")[-1].strip()

    text = text.replace("Score", "score")
    if "score:" in text:
        text = text.split("score:")[-1].strip()
    if "answer:" in text:
        text = text.split("answer:")[0].strip()

    return text


def main():
    """Run the main inference function."""
    # load qa dataset
    dataset = datasets.load_dataset(
        "json",
        data_files={"test": args.dataset_path},
        cache_dir=args.cache_dir,
    )
    testset = dataset["test"]
    if args.start_index is None:
        args.start_index = 0
    if args.end_index is None:
        args.end_index = len(testset)
    data_size = args.end_index - args.start_index
    testset = testset.select(range(args.start_index, args.end_index))
    print(
        f"Load dataset of size #{data_size} | ",
        f"Processing [{args.start_index} - {args.end_index}]",
    )

    # initialize tokenizer and model
    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name_or_path
    if "llama" in args.model_name_or_path:
        from transformers import LlamaTokenizer, LlamaForCausalLM
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_name_or_path)
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    if args.lora_weights_path is not None:
        model = PeftModel.from_pretrained(model, args.lora_weights_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # tokenization & generation kwargs
    tokenization_kwargs = {"return_tensors": "pt"}

    gene_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
    }
    if args.temperature is not None:
        gene_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        gene_kwargs["top_p"] = args.top_p

    # collect model outputs
    outset = []
    for start_idx in range(args.start_index, args.end_index, args.batch_size):
        end_idx = min(start_idx + args.batch_size, args.end_index)
        batch_inputs = testset["input"][start_idx:end_idx]
        batch_outputs = testset["output"][start_idx:end_idx]
        # print('batch_outputs:', batch_outputs)

        input_dict = tokenizer(batch_inputs, padding=True, truncation=True, **tokenization_kwargs)
        input_dict = {k: v.to(model.device) for k, v in input_dict.items()}
        output_dict = model.generate(
            input_ids=input_dict["input_ids"],
            attention_mask=input_dict["attention_mask"],
            **gene_kwargs,
        )

        predictions = tokenizer.batch_decode(output_dict, skip_special_tokens=True)  # list[str]
        pred_answers = [clean_output(p, prefix=inp) for p, inp in zip(predictions, batch_inputs)]
        for inp, out, preds, pred_ans in zip(batch_inputs, batch_outputs, predictions, pred_answers):
            out_ex = {
                "input": inp,
                "output": out,
                "predictions": [preds],
                "pred_answers": [pred_ans],
            }
            outset.append(out_ex)

        if (start_idx + 1) % args.report_steps == 0 or end_idx == args.end_index:
            print(f"\n\nSample {end_idx}")
            print(f"Input: \n{batch_inputs[-1]}\n", "-" * 20)
            print(f"Output: \n{batch_outputs[-1]}\n", "-" * 20)
            print("Candidates: ")
            for p, a in zip(predictions, pred_answers):
                print(f"{p} | {a}")
            print("=" * 50)

    write_dataset(path=args.output_path, dataset=outset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data i/o path
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    # huggingface model and tokenizer
    parser.add_argument("--model_name_or_path", type=str, default="google/flan-t5-xl")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--lora_weights_path", type=str, default=None)
    # tokenization & generation kwargs
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Maximum length of the sequence to be generated.",
    )
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=None)  # 0.8
    parser.add_argument("--top_p", type=float, default=None)  # 0.95
    # logging
    parser.add_argument("--report_steps", type=int, default=500)
    parser.add_argument("--start_index", type=int, default=None)
    parser.add_argument("--end_index", type=int, default=None)
    # batching
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    args = parser.parse_args()
    main()
