# Convert whisper into smaller model using layer pruning
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from time import time
from collections import defaultdict
from tqdm import tqdm
from itertools import islice

import torch
from transformers import (
    WhisperProcessor,
    GenerationConfig,
    WhisperForConditionalGeneration,
    WhisperTokenizer
)
from evaluate import load
from datasets import load_dataset
from pydub import AudioSegment
import numpy as np


def map_to_pred(batch, model, processor):
    audio = batch["audio"]
    audio = AudioSegment(audio["bytes"], frame_rate=16000, sample_width=2, channels=1)
    audio = np.array(audio.get_array_of_samples(), dtype=np.int16) / 32768

    # input_features = processor(
    #     audio, sampling_rate=16000, return_tensors="pt"
    # ).input_features
    input_features = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="russian", task="transcribe")

    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("cuda"), forced_decoder_ids=forced_decoder_ids)[0]
    transcription = processor.decode(predicted_ids)
    return processor.tokenizer._normalize(transcription)


TEACHER_CKPT = "openai/whisper-medium"  # 22 decoder layers
# STUDENT_MODEL = "/raid/nanosemantics/vyaznikov/itmo_hw/itmo_compression_course/hw3/whisper_student"
STUDENT_MODEL = "openai/whisper-base"
DECODER_LAYERS = 6
SAVE_DIR = "."
CACHE_DIR = "."
NUM_EXAMPLES = 50  # for validation
DATASET_DIR = "/raid/nanosemantics/vyaznikov/itmo_hw"


teacher_model = WhisperForConditionalGeneration.from_pretrained(
    TEACHER_CKPT, cache_dir=CACHE_DIR
).to("cuda")
student_model = WhisperForConditionalGeneration.from_pretrained(
    STUDENT_MODEL,
    cache_dir=CACHE_DIR,
).to("cuda")
processor = WhisperProcessor.from_pretrained(TEACHER_CKPT, cache_dir=CACHE_DIR)


def prepare_student_model():
    teacher_config = teacher_model.config
    teacher_layers = teacher_config.decoder_layers
    student_config = teacher_config
    student_config.decoder_layers = DECODER_LAYERS

    mapping = [
        0,
        1,
        4,
        8,
        16,
        23,
    ]  # mapping 8 teacher decoder layers to student model
    assert DECODER_LAYERS == len(mapping)

    student_model = WhisperForConditionalGeneration(student_config)

    # copy layers
    info = student_model.load_state_dict(teacher_model.state_dict(), strict=False)

    # make sure entire encoder is copied
    for s, t in zip(
        student_model.model.encoder.parameters(),
        teacher_model.model.encoder.parameters(),
    ):
        assert torch.equal(s.data, t.data)

    # copy decoder layers
    # has to be strict match: <All keys matched successfully>
    layers_to_copy = torch.nn.ModuleList(
        [teacher_model.model.decoder.layers[i] for i in mapping]
    )
    student_model.model.decoder.layers.load_state_dict(layers_to_copy.state_dict())

    # save model
    student_model.save_pretrained(SAVE_DIR)

    # also save processor, generation config and tokenizer
    processor = WhisperProcessor.from_pretrained(TEACHER_CKPT, cache_dir=CACHE_DIR)
    processor.save_pretrained(SAVE_DIR)
    generation_config = GenerationConfig.from_pretrained(
        TEACHER_CKPT, cache_dir=CACHE_DIR
    )
    generation_config.save_pretrained(SAVE_DIR)
    tokenizer = WhisperTokenizer.from_pretrained(TEACHER_CKPT, cache_dir=CACHE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)


def validate_models():
    validation_dataset = load_dataset(
        "bond005/sberdevices_golos_10h_crowd",
        "clean",
        split="validation",
        cache_dir=DATASET_DIR,
    )
    validation_dataset_slice = islice(validation_dataset, NUM_EXAMPLES)

    res = defaultdict(list)

    t = time()
    for el in tqdm(validation_dataset_slice):
        res["reference"].append(processor.tokenizer._normalize(el["transcription"]))
        res["prediction_teacher"].append(map_to_pred(el, teacher_model, processor))

    t = ((time() - t) / NUM_EXAMPLES) * 1000
    print(f"Teacher: avg. time on example: {t}")

    validation_dataset_slice = islice(validation_dataset, NUM_EXAMPLES)
    t = time()
    for el in tqdm(validation_dataset_slice):
        # res["reference"].append(processor.tokenizer._normalize(el["transcription"]))
        res["prediction_student"].append(map_to_pred(el, student_model, processor))

    t = ((time() - t) / NUM_EXAMPLES) * 1000
    print(f"Student: avg. time on example: {t}")

    wer = load("wer")
    print(
        f"WER on teacher = {100 * wer.compute(references=res['reference'], predictions=res['prediction_teacher'])}"
    )
    print(
        f"WER on student = {100 * wer.compute(references=res['reference'], predictions=res['prediction_student'])}"
    )

    # print(res['prediction_student'])


if __name__ == "__main__":
    # prepare_student_model()
    validate_models()
