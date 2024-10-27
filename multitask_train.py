import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import ast
import argparse
import logging
import torch
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import EvalPrediction
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from torch import nn
import evaluate
from process_inputs import SEP_TOKEN
from query import clean_output
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class FlanT5WithClassificationHead(nn.Module):
    def __init__(self, model_name, embed_dim, num_classes):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=self.config)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None, input_indices_list=None, **kwargs):
        seq2seq_output = self.seq2seq_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **kwargs
        )
        encoder_last_hidden_state = seq2seq_output.encoder_last_hidden_state
        # Initializes the tensor list of stored features
        logits_list = []

        for i in range(input_indices_list.size(0)):
            start_question = input_indices_list[i, 0].item()
            end_question = input_indices_list[i, 1].item()

            start_pseudo_answer = input_indices_list[i, 1].item()
            end_pseudo_answer = input_indices_list[i, 2].item()

            start_context = input_indices_list[i, 2].item()

            # Extract feature
            question_features = encoder_last_hidden_state[i, start_question:end_question, :].unsqueeze(0)
            pseudo_answer_features = encoder_last_hidden_state[i, start_pseudo_answer:end_pseudo_answer, :].unsqueeze(0)
            context_features = encoder_last_hidden_state[i, start_context:, :].unsqueeze(0)

            pseudo_output, _ = self.cross_attention(question_features, pseudo_answer_features, pseudo_answer_features,
                                                    attn_mask=None)
            # print('pseudo_output.shape:',pseudo_output.shape)
            pseudo_pooled_output = self.global_avg_pool(pseudo_output.transpose(1, 2))
            pseudo_pooled_output = torch.squeeze(pseudo_pooled_output, -1)
            pseudo_logits = self.fc(pseudo_pooled_output)
            logits_list.append(pseudo_logits)
            context_output, _ = self.cross_attention(question_features, context_features, context_features,
                                                     attn_mask=None)
            context_pooled_output = self.global_avg_pool(context_output.transpose(1, 2))
            context_pooled_output = torch.squeeze(context_pooled_output, -1)
            context_logits = self.fc(context_pooled_output)
            logits_list.append(context_logits)
        class_logit = torch.stack(logits_list).squeeze(1)
        # print(class_logit, class_logit.shape)

        return {
            "loss": seq2seq_output.loss,
            "logits": seq2seq_output.logits,
            "class_logit": class_logit,
        }

    def generate(self, **kwargs):
        return self.seq2seq_model.generate(**kwargs)


def main():
    # Load config, tokenizer, and model
    config = AutoConfig.from_pretrained(
        args.config_name if (args.config_name is not None) else args.model_name,
        cache_dir=args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if (args.tokenizer_name is not None) else args.model_name,
        cache_dir=args.cache_dir,
    )
    model = FlanT5WithClassificationHead(args.model_name, args.hidden_dimension, args.num_classes)

    # Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"],
        task_type="SEQ_2_SEQ_LM",
    )
    model.seq2seq_model = get_peft_model(model.seq2seq_model, lora_config)

    # Load train, validation, and test data
    data_files = {}
    if args.train_data_path:
        data_files["train"] = args.train_data_path
        extension = args.train_data_path.split(".")[-1]
    if args.eval_data_path:
        data_files["validation"] = args.eval_data_path
        extension = args.eval_data_path.split(".")[-1]
    if args.test_data_path:
        data_files["test"] = args.test_data_path
        extension = args.test_data_path.split(".")[-1]
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=args.cache_dir,
    )

    # Column names to remove after preprocessing
    train_column_names, eval_column_names, test_column_names = None, None, None
    if "train" in raw_datasets:
        train_column_names = raw_datasets["train"].column_names
    if "validation" in raw_datasets:
        eval_column_names = raw_datasets["validation"].column_names
    if "test" in raw_datasets:
        test_column_names = raw_datasets["test"].column_names

    # Tokenization config
    padding = "max_length" if args.pad_to_max_length else False
    max_seq_length = min(args.max_seq_length, config.n_positions)
    max_answer_length = args.max_answer_length

    # Training args
    train_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "evaluation_strategy": args.evaluation_strategy,
        "eval_steps": args.eval_steps,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
    }
    train_kwargs = {k: v for k, v in train_kwargs.items() if v is not None}
    training_args = Seq2SeqTrainingArguments(
        do_train=True,
        do_eval=True,
        do_predict=True,
        predict_with_generate=True,
        report_to="none",
        **train_kwargs,
    )

    # Preprocessing
    def preprocess_function(examples):
        inputs = examples["input"]
        targets = examples["output"]
        model_inputs = tokenizer(
            inputs,
            max_length=max_seq_length,
            padding=padding,
            truncation=True,
        )
        labels = tokenizer(
            text_target=targets,
            max_length=max_answer_length,
            padding=padding,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset, eval_dataset, predict_dataset = None, None, None
    if "train" in raw_datasets:
        train_examples = raw_datasets["train"]
        train_dataset = train_examples.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=train_column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    if "validation" in raw_datasets:
        eval_examples = raw_datasets["validation"]
        eval_dataset = eval_examples.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=eval_column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
    if "test" in raw_datasets:
        predict_examples = raw_datasets["test"]
        predict_dataset = predict_examples.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=test_column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on test dataset",
        )
    # Data collator
    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model.seq2seq_model,
        label_pad_token_id=label_pad_token_id,
    )
    # Custom metric
    metric = evaluate.load("exact_match")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        preds[preds < 0] = 0
        try:
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        except Exception as e:
            # print('max:', tokenizer.vocab_size)  # 32100
            print(e)
        decoded_preds = [clean_output(dp) for dp in decoded_preds]
        p.label_ids[p.label_ids < 0] = 0
        decoded_labels = tokenizer.batch_decode(p.label_ids, skip_special_tokens=True)
        decoded_preds = [clean_output(dp) for dp in decoded_preds]
        decoded_labels = [clean_output(dl) for dl in decoded_labels]
        seq2seq_metrics = metric.compute(predictions=decoded_preds, references=decoded_labels)

        return {
            **seq2seq_metrics,
        }

    # Custom Trainer
    class CustomSeq2SeqTrainer(Seq2SeqTrainer):
        def restore_input(self, inputs_labels, decoder_input):
            output_tokens = tokenizer.convert_ids_to_tokens(inputs_labels)
            try:
                start_idx = output_tokens.index(SEP_TOKEN)
            except Exception as e:
                print('output_tokens:', output_tokens)
            # Find the end location of classification_labels
            end_idx = output_tokens.index('</s>')
            classification_labels_str = ''.join(output_tokens[start_idx:end_idx])
            classification_labels_str = classification_labels_str.replace('▁', ' ')
            try:
                classification_labels = ast.literal_eval(classification_labels_str.split(': ')[1])
            except SyntaxError as e:
                classification_labels = [0, 0]
                print('output_tokens:', output_tokens)
                print(f"Error parsing {classification_labels_str}: {e}")
            mask = torch.arange(inputs_labels.size(0), device='cuda:0') > start_idx
            mask[start_idx] = False
            inputs_labels = torch.where(mask, 0, inputs_labels)
            inputs_labels[start_idx] = 1
            mask = torch.arange(decoder_input.size(0),
                                device='cuda:0') > start_idx + 1
            mask[start_idx + 1] = False
            decoder_input = torch.where(mask, 0, decoder_input)
            decoder_input[start_idx + 1] = 1
            return inputs_labels, decoder_input, classification_labels

        def compute_loss(self, model, inputs, sigma=0.2, return_outputs=False):
            input_indices_list = []
            classification_labels_list = []
            for batch_idx in range(inputs["input_ids"].size(0)):
                input_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][batch_idx])
                sep_input_indices = [i for i, token in enumerate(input_tokens) if token == SEP_TOKEN]
                inputs_labels, decoder_input, classification_labels = self.restore_input(inputs["labels"][batch_idx],
                                                                                         inputs["decoder_input_ids"][
                                                                                             batch_idx])
                inputs["labels"][batch_idx] = inputs_labels
                inputs["decoder_input_ids"][batch_idx] = decoder_input
                classification_labels_list.append(classification_labels)
                input_indices_list.append(sep_input_indices)
            inputs['input_indices_list'] = torch.tensor(input_indices_list, device='cuda:0')
            outputs = model(**inputs)
            seq2seq_loss = outputs["loss"]
            classification_loss_fn = nn.CrossEntropyLoss()
            classification_labels = torch.tensor(classification_labels_list, device='cuda:0')
            classification_labels = classification_labels.flatten()
            classification_loss = classification_loss_fn(outputs["class_logit"], classification_labels)
            total_loss = (1 - sigma) * seq2seq_loss + sigma * classification_loss
            return (total_loss, outputs) if return_outputs else total_loss

        def prediction_step(
                self,
                model: nn.Module,
                inputs: Dict[str, Union[torch.Tensor, Any]],
                prediction_loss_only: bool,
                ignore_keys: Optional[List[str]] = None,
        ):
            if not self.args.predict_with_generate or prediction_loss_only:
                return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

            inputs = self._prepare_inputs(inputs)
            gen_kwargs = {
                "max_length": self.model.config.max_length if hasattr(self.model, 'config') and hasattr(
                    self.model.config, 'max_length') else self.args.max_length,
                "num_beams": self.model.config.num_beams,
            }

            generated_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )
            # torch.cuda.empty_cache()
            loss = None
            if "labels" in inputs:
                with torch.no_grad():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    outputs["generated_tokens"] = generated_tokens
            else:
                loss = None

            if prediction_loss_only:
                return (loss, None, None)

            if "labels" in inputs:
                labels = inputs["labels"]
            else:
                labels = None

            return (loss, generated_tokens, labels)

    # Initialize Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if "train" in raw_datasets else None,
        eval_dataset=eval_dataset if "validation" in raw_datasets else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()

        # Save LoRA weights separately
        output_path = os.path.join(args.output_dir, "lora")
        os.makedirs(output_path, exist_ok=True)
        model.seq2seq_model.save_pretrained(output_path)

        metrics = train_result.metrics
        max_train_examples = (
            args.max_train_examples if args.max_train_examples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_examples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            max_length=args.max_answer_length,
            num_beams=args.num_beams if args.num_beams is not None else training_args.generation_num_beams,
            metric_key_prefix="eval"
        )
        max_eval_examples = (
            args.max_eval_examples if args.max_eval_examples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_examples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(predict_dataset)
        metrics = results.metrics
        max_predict_examples = (
            args.max_predict_examples if args.max_predict_examples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_examples, len(predict_dataset))
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # hf name
    parser.add_argument("--model_name", type=str, default="google/flan-t5-xl")
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--overwrite_cache", action="store_true")

    # data path
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--test_data_path", type=str, default=None)

    # preprocess config
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length (by tokens) for inputs.",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=64,
        help="Maximum sequence length (by tokens) for answers.",
    )
    parser.add_argument("--pad_to_max_length", action="store_true")
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)

    # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments
    # training config
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--hidden_dimension", type=int, default=2048)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--distribute_model", action="store_true")
    # evaluation & save model
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="epoch",
        choices=["epoch", "steps", "no"],
    )
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument(
        "--save_strategy", type=str, default="epoch", choices=["epoch", "steps", "no"]
    )
    parser.add_argument("--save_steps", type=int, default=None)
    # generation config
    parser.add_argument("--num_beams", type=int, default=1)
    # train/eval/test mode
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_eval_examples", type=int, default=None)
    parser.add_argument("--max_predict_examples", type=int, default=None)

    args = parser.parse_args()

    main()
