import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# Run teacher student distillation for Whisper model
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    set_seed,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset, DatasetDict, interleave_datasets, IterableDatasetDict
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import transformers
import argparse
import datasets
import evaluate
import string
from accelerate import Accelerator
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch
from tqdm.auto import tqdm
import numpy as np
from pydub import AudioSegment


def load_streaming_dataset(dataset_name, dataset_config_name, split, **kwargs):
    if "+" in split:
        # load multiple splits separated by the `+` symbol *with* streaming mode
        dataset_splits = [
            load_dataset(
                dataset_name,
                dataset_config_name,
                split=split_name,
                streaming=True,
                **kwargs,
            )
            for split_name in split.split("+")
        ]
        # interleave multiple splits to form one dataset
        interleaved_dataset = interleave_datasets(dataset_splits)
        return interleaved_dataset
    else:
        # load a single split *with* streaming mode
        dataset = load_dataset(
            dataset_name, dataset_config_name, split=split, streaming=True, **kwargs
        )
        return dataset


def decode_audio(example):
    byte_audio = example["audio"]["bytes"]
    audio = AudioSegment(byte_audio, frame_rate=16000, sample_width=2, channels=1)
    audio = np.array(audio.get_array_of_samples(), dtype=np.int16) / 32768

    example["audio"] = {"array": audio, "sampling_rate": 16000}
    return example


def train(args, accelerator: Accelerator):
    # load dataset in streaming mode
    raw_datasets = IterableDatasetDict()
    raw_datasets["train"] = load_streaming_dataset(
        args.train_dataset_name,
        args.train_dataset_config_name,
        split=args.train_split_name,
        cache_dir=args.data_cache_dir,
    )
    raw_datasets["validation"] = load_streaming_dataset(
        args.validation_dataset_name,
        args.validation_dataset_config_name,
        split=args.validation_split_name,
        cache_dir=args.data_cache_dir,
    )

    # raw_datasets = raw_datasets.remove_columns(["file", "speaker_id", "chapter_id", "id"])
    # raw_datasets = raw_datasets.rename_columns({'audio': 'audio', 'text': 'text'})

    assert args.audio_column in raw_datasets["train"].column_names
    assert args.text_column in raw_datasets["train"].column_names

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            raw_datasets["train"] = raw_datasets["train"].take(args.max_train_samples)
        if args.max_val_samples is not None:
            raw_datasets["validation"] = raw_datasets["validation"].take(
                args.max_val_samples
            )

    student_config = AutoConfig.from_pretrained(
        args.student_model_name_or_path, cache_dir=args.student_cache_dir
    )
    teacher_config = AutoConfig.from_pretrained(
        args.teacher_model_name_or_path, cache_dir=args.teacher_cache_dir
    )

    # assuming student and teacher uses same feature extractor, tokenizer and processor
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.teacher_model_name_or_path, cache_dir=args.teacher_cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model_name_or_path, cache_dir=args.teacher_cache_dir#, language="russian"
    )
    processor = AutoProcessor.from_pretrained(
        args.teacher_model_name_or_path, cache_dir=args.teacher_cache_dir
    )
    processor.tokenizer.set_prefix_tokens(language="russian", task="transcribe")

    # make sure decoder_start_token_id is defined for both
    assert teacher_config.decoder_start_token_id is not None
    assert student_config.decoder_start_token_id is not None

    # We need to set the language and task ids for previously multilingual checkpoints, default is English and transcribe
    # Set to None if the model is not multilingual
    student_config.forced_decoder_ids = None
    # tokenizer.get_decoder_prompt_ids(language=args.language, task=args.task, no_timestamps=True)
    teacher_config.forced_decoder_ids = None
    # tokenizer.get_decoder_prompt_ids(language=args.language, task=args.task, no_timestamps=True)
    student_config.suppress_tokens = []
    teacher_config.suppress_tokens = []

    student_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.student_model_name_or_path, 
        config=student_config, 
        cache_dir=args.student_cache_dir
    )
    teacher_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.teacher_model_name_or_path,
        config=teacher_config,
        cache_dir=args.teacher_cache_dir,
    )

    accelerator.print(
        f"Loaded the model on device: student: {student_model.device}, teacher:{teacher_model.device}, accelerator:{accelerator.device}"
    )

    # freeze teacher model
    for p in teacher_model.parameters():
        p.requires_grad = False

    if args.freeze_encoder:
        accelerator.print("Freezing encoder")
        student_model.freeze_encoder()
        student_model.model.encoder.gradient_checkpointing = False
    
    def is_transcription_valid(transcription):
        return bool(transcription)

    # Resample speech dataset: so we just need to set the correct target sampling rate
    with accelerator.main_process_first():
        # raw_datasets = raw_datasets.cast_column(args.audio_column,
        #                                         datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate))
        raw_datasets = raw_datasets.map(decode_audio)
        raw_datasets = raw_datasets.filter(
            is_transcription_valid, input_columns=["transcription"]
        )

    # Preprocessing the raw_datasets, need to read the audio files as arrays and tokenize the targets.
    # might need to change the normalizer depending on language and task
    normalizer = BasicTextNormalizer()

    def prepare_dataset(batch):
        # process audio
        sample = batch[args.audio_column]
        # compute log-Mel input features from input audio array
        batch["input_features"] = processor.feature_extractor(
            sample["array"], sampling_rate=sample["sampling_rate"]
        ).input_features[0]
        # process audio length
        batch["input_length"] = len(sample["array"]) / sample["sampling_rate"]
        # process
        transcription = batch[args.text_column]
        if not args.keep_case:
            transcription = transcription.lower()
        if not args.keep_punctuation:
            transcription = normalizer(transcription).strip()
        batch["labels"] = processor.tokenizer(transcription).input_ids
        return batch

    with accelerator.main_process_first():
        vectorized_datasets = raw_datasets.map(
            prepare_dataset, remove_columns=raw_datasets["train"].column_names
        )

    # filter training data with inputs longer than max_input_length
    def is_audio_in_length_range(length):
        # return min_input_length <= length <= max_input_length
        return args.min_duration_in_seconds <= length <= args.max_duration_in_seconds

    with accelerator.main_process_first():
        vectorized_datasets = vectorized_datasets.filter(
            is_audio_in_length_range, input_columns=["input_length"]
        )

    @dataclass
    class DataCollatorForSeq2SeqWithPadding:
        processor: Any

        def __call__(
            self, features: List[Union[Dict[str, torch.Tensor], Dict[str, Any]]]
        ) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lengths and need different padding methods
            # first treat the audio inputs by simply returning torch tensors
            input_features = [
                {"input_features": feature["input_features"]} for feature in features
            ]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )

            # then pad the labels
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )
            # replace -100 in labels by tokenizer.pad_token_id to ignore padding in loss
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            # if bos token is appended in previous step, remove it here as it's appended again in the forward pass
            if (
                (labels[:, 0] == self.processor.tokenizer.bos_token_id)
                .all()
                .cpu()
                .item()
            ):
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    data_collator = DataCollatorForSeq2SeqWithPadding(processor=processor)

    # now define data loaders
    train_dataloader = torch.utils.data.DataLoader(
        vectorized_datasets["train"],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        vectorized_datasets["validation"],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # define optimizer
    optimizer = torch.optim.AdamW(
        list(student_model.parameters()), lr=args.learning_rate
    )

    # scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps,
    )

    # accelerator setup for distributed training, this handles all the device mapping, gradient accumulation, fp16 training etc.
    # add eval_dataloader to accelerator.prepare for distributed evaluation
    (
        student_model,
        teacher_model,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        student_model, teacher_model, optimizer, train_dataloader, lr_scheduler
    )
    accelerator.print(
        f"Distributed the model on device: student: {student_model.device}, teacher:{teacher_model.device}, accelerator:{accelerator.device}"
    )

    global_step = 0  # global step for logging
    total_loss = 0  # total loss before each eval
    total_kl_loss = 0  # total kl loss before each eval
    total_ce_loss = 0  # total ce loss before each eval

    if args.resume_from_checkpoint is not None:
        accelerator.print(f"Loading checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        steps_completed = int(args.resume_from_checkpoint.split("-")[-1])
        global_step += steps_completed
        train_dataloader = accelerator.skip_first_batches(
            train_dataloader, steps_completed
        )

    # load metric
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    all_punctuations = list(string.punctuation.replace("'", ""))

    def compute_metrics(preds, labels):
        # replace padded labels by padding token
        for idx in range(len(labels)):
            labels[idx][labels[idx] == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

        pred_str = [_.strip() for _ in pred_str]
        label_str = [_.strip() for _ in label_str]

        # space punctuation for orthographic WER
        spaced_pred_str = [
            pred_str[i].replace(punctuation, "")
            for punctuation in all_punctuations
            for i in range(len(pred_str))
        ]
        spaced_label_str = [
            label_str[i].replace(punctuation, "")
            for punctuation in all_punctuations
            for i in range(len(label_str))
        ]

        # compute WER
        wer_ortho = 100 * wer_metric.compute(
            predictions=spaced_pred_str, references=spaced_label_str
        )
        cer_ortho = 100 * cer_metric.compute(
            predictions=spaced_pred_str, references=spaced_label_str
        )
        accelerator.print(
            f"\nspaced_pred_str: {[_ for i, _ in enumerate(spaced_pred_str) if i < 3]}, \nspaced_label_str: {[_ for i, _ in enumerate(spaced_label_str) if i < 3]}"
        )
        # normalize everything and re-compute the WER
        norm_pred_str = [normalizer(pred) for pred in pred_str]
        norm_label_str = [normalizer(label) for label in label_str]
        # filtering step to only evaluate the samples that correspond to non-zero normlized references:
        norm_pred_str = [
            norm_pred_str[i]
            for i in range(len(norm_label_str))
            if len(norm_label_str[i]) > 0
        ]
        norm_label_str = [
            norm_label_str[i]
            for i in range(len(norm_label_str))
            if len(norm_label_str[i]) > 0
        ]
        accelerator.print(
            f"\nnorm_pred_str: {[_ for i, _ in enumerate(norm_pred_str) if i < 3]}, \nnorm_label_str: {[_ for i, _ in enumerate(norm_label_str) if i < 3]}"
        )

        wer = 100 * wer_metric.compute(
            predictions=norm_pred_str, references=norm_label_str
        )
        cer = 100 * cer_metric.compute(
            predictions=norm_pred_str, references=norm_label_str
        )

        return (
            {"wer": wer, "wer_ortho": wer_ortho, "cer": cer, "cer_ortho": cer_ortho},
            pred_str,
            label_str,
        )

    # save feature extractor, tokenizer, config and generation config
    with accelerator.main_process_first():
        output_dir = args.output_dir
        feature_extractor.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        student_config.save_pretrained(output_dir)
        teacher_config.save_pretrained(output_dir)

    progress_bar = tqdm(
        range(global_step, args.train_steps), disable=not accelerator.is_main_process
    )
    # define training step
    while global_step < args.train_steps:
        student_model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(student_model):
                # forward pass
                outputs = student_model(**batch)
                ce_loss = outputs.loss
                logits = outputs.logits
                with torch.no_grad():
                    teacher_logits = teacher_model(**batch).logits
                # compute kl loss
                kl_loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(logits / args.temperature, dim=-1),
                    torch.nn.functional.softmax(
                        teacher_logits / args.temperature, dim=-1
                    ),
                    reduction="batchmean",
                ) * (args.temperature**2)
                # compute total loss
                loss = args.alpha_ce * ce_loss + args.alpha_distil * kl_loss

                total_kl_loss += kl_loss.detach().item()
                total_ce_loss += ce_loss.detach().item()
                total_loss += loss.detach().item()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                progress_bar.update(1)

                # log metrics
                eval_metrics = {}
                eval_preds = []
                eval_labels = []

                if global_step % args.eval_steps == 0:
                    student_model.eval()
                    valid_loss = 0
                    # validation_progress_bar = tqdm(range(0, len(eval_dataloader)), disable=not accelerator.is_main_process)
                    for batch in eval_dataloader:
                        with torch.no_grad():
                            batch.to(accelerator.device)
                            references = batch.labels
                            if not args.predict_without_generate:
                                # accelerator.print("\nPredicting with generate")
                                # for modules wrapped in DataParallel or DistributedDataParallel, we need to use .module to access the underlying module
                                if accelerator.num_processes > 1:
                                    # accelerator.print("Distributed eval")
                                    predictions = student_model.module.generate(
                                        batch.input_features
                                    )
                                else:
                                    predictions = student_model.generate(
                                        batch.input_features
                                    )
                            else:
                                # accelerator.print("\nPredicting without generate")
                                outputs = student_model(**batch)
                                valid_loss += outputs.loss.detach().item()
                                pred_logits = outputs.logits
                                predictions = pred_logits.argmax(-1)

                        # accelerator.print("Before gather")
                        # accelerator.print(f"len of predictions: {len(predictions)}, len of references: {len(references)}")
                        # accelerator.print(f"All types for gather has to be tensor: \ntype of predictions: {type(predictions)}, type of references: {type(references)}")
                        predictions, references = accelerator.gather_for_metrics(
                            (predictions, references)
                        )
                        # accelerator.print("After gather")
                        # accelerator.print(f"len of predictions: {len(predictions)}, len of references: {len(references)}")

                        ###########################
                        # convert any token after after first tokenizer.eos_token_id to eos_token_id
                        for idx, pred in enumerate(predictions):
                            first_eos_token_idx = (
                                pred == tokenizer.eos_token_id
                            ).nonzero(as_tuple=True)[0]
                            if len(first_eos_token_idx) > 0:
                                predictions[
                                    idx, first_eos_token_idx[0] + 1 :
                                ] = tokenizer.eos_token_id
                        ###########################

                        eval_preds.extend(predictions)
                        eval_labels.extend(references)
                        # validation_progress_bar.update(1)

                    accelerator.print(
                        f"\npredictions: {eval_preds[:3]}, \nreferences: {eval_preds[:3]}"
                    )
                    accelerator.print(
                        f"\nlen(eval_preds): {len(eval_preds)}, \nlen(eval_labels): {len(eval_labels)}"
                    )
                    eval_metrics, eval_preds, eval_labels = compute_metrics(
                        eval_preds, eval_labels
                    )
                    train_loss = total_loss / (
                        args.eval_steps
                        * args.per_device_train_batch_size
                        * accelerator.num_processes
                    )
                    train_kl_loss = total_kl_loss / (
                        args.eval_steps
                        * args.per_device_train_batch_size
                        * accelerator.num_processes
                    )
                    train_ce_loss = total_ce_loss / (
                        args.eval_steps
                        * args.per_device_train_batch_size
                        * accelerator.num_processes
                    )

                    accelerator.print(
                        f"Step: {global_step}, Train Loss: {train_loss}, Train KL Loss: {train_kl_loss}, Train CE Loss: {train_ce_loss}, \
                                       Eval WER: {eval_metrics['wer']}, Eval WER Ortho: {eval_metrics['wer_ortho']}, Eval CER: {eval_metrics['cer']}, \
                                      Eval CER Ortho: {eval_metrics['cer_ortho']}"
                    )
                    accelerator.log(
                        {
                            "cer": eval_metrics["cer"],
                            "wer": eval_metrics["wer"],
                            "train_loss": train_loss,
                            "train_kl_loss": train_kl_loss,
                            "train_ce_loss": train_ce_loss,
                            #  "eval_loss": valid_loss / (len(eval_dataloader))
                        },
                        step=global_step
                    )

                    # output_dir = os.path.join(
                    #     args.output_dir, f"checkpoint-{global_step}"
                    # )
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint"
                    )
                    # accelerator.save(student_model.state_dict(), output_dir)
                    accelerator.save_state(output_dir)
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(student_model)
                    unwrapped_model.save_pretrained(
                        output_dir,
                        save_function=accelerator.save,
                        is_main_process=accelerator.is_main_process,
                    )
                    total_loss = 0
                    total_kl_loss = 0
                    total_ce_loss = 0
                    student_model.train()


def main():
    # accelerate launch student-teacher-distillation-streaming.py
    # --freeze_encoder
    # --keep_punctuation
    # --keep_case
    # --teacher_model_name_or_path openai/whisper-large-v2
    # --student_model_name_or_path large-v2-2
    # --student_cache_dir large-v2-8
    # --output_dir whisper-large-v2-8-en-cv
    # --data_cache_dir commonvoice
    # --teacher_cache_dir cache
    # --student_cache_dir large-v2-8-en-cv
    # --text_column sentence
    # --train_dataset_name mozilla-foundation/common_voice_13_0
    # --train_dataset_config_name en
    # --train_split_name train
    # --validation_dataset_name mozilla-foundation/common_voice_13_0
    # --validation_dataset_config_name en
    # --validation_split_name test
    # --max_val_samples 2000
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher_model_name_or_path", type=str, default="openai/whisper-large-v2"
    )
    parser.add_argument(
        "--student_model_name_or_path", type=str, default="distil-whisper/large-v2-8"
    )
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)  # 4
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha_ce", type=float, default=0.5)
    parser.add_argument("--alpha_distil", type=float, default=0.5)
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--task", type=str, default="transcribe")
    parser.add_argument("--train_steps", type=int, default=2500)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4
    )  # increase by 2x for every 2x decrease in batch size 64
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_cache_dir", type=str, default="data/cache")
    parser.add_argument("--teacher_cache_dir", type=str, default="model/cache")
    parser.add_argument("--student_cache_dir", type=str, default="model/cache")
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--audio_column", type=str, default="audio")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--max_duration_in_seconds", type=float, default=30)
    parser.add_argument("--min_duration_in_seconds", type=float, default=1)
    parser.add_argument("--keep_case", action="store_true")
    parser.add_argument("--keep_punctuation", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--predict_without_generate", action="store_true")
    parser.add_argument("--train_dataset_name", type=str, default="librispeech_asr")
    parser.add_argument("--train_dataset_config_name", type=str, default="all")
    parser.add_argument(
        "--train_split_name",
        type=str,
        default="train.clean.100+train.clean.360+train.other.500",
    )
    parser.add_argument(
        "--validation_dataset_name", type=str, default="librispeech_asr"
    )
    parser.add_argument("--validation_dataset_config_name", type=str, default="all")
    parser.add_argument("--validation_split_name", type=str, default="validation.clean")

    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    if (
        args.teacher_model_name_or_path is None
        or args.student_model_name_or_path is None
    ):
        raise ValueError(
            "teacher_model_name_or_path and student_model_name_or_path cannot be None"
        )

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_dir=args.output_dir,
    )

    # have only one message per logs of transformers or datasets, so logging verbosity INFO only for the main process
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_info()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # establish trackers for logging
    track_config = {
        "lr": args.learning_rate,
        "train_batch_size": args.per_device_train_batch_size,
        "eval_batch_size": args.per_device_eval_batch_size,
        "seed": args.seed,
        "train_steps": args.train_steps,
    }
    accelerator.init_trackers("runs", track_config)
    train(args, accelerator)
    accelerator.end_training()


if __name__ == "__main__":
    main()
