from typing import Iterable, Iterator

import more_itertools
import torch
from tango.common.tqdm import Tqdm
from transformers import AutoModelForMultipleChoice, AutoTokenizer, AutoModelForQuestionAnswering

from ludwig.models.model import ModelForEvaluation
from ludwig.tasks import GenerationTask, MCTask, QATask, ClassificationTask, PairClassificationTask
from ludwig.utilities import get_best_spans


@ModelForEvaluation.register("hf")
class HFAutoModelForEvaluation(ModelForEvaluation):
    VERSION = "003"

    def __init__(self, pretrained_model_name_or_path: str):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

    def do_multiple_choice(
        self,
        task: MCTask,
        instances: Iterable[MCTask.Instance],
        *,
        batch_size: int = 32
    ) -> Iterator[MCTask.InstanceResult]:
        # There is no Huggingface pipeline for this.
        model = AutoModelForMultipleChoice.from_pretrained(self.pretrained_model_name_or_path).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        instances = Tqdm.tqdm(instances, desc="Processing instances")
        with torch.inference_mode():
            for batch in more_itertools.chunked(instances, batch_size):
                texts = []
                labels = []
                for instance in batch:
                    assert len(instance.answer_choices) == task.number_of_choices
                    context = instance.context or ""
                    texts.extend(
                        (f"{context} {instance.question}".strip(), choice.strip())
                        for choice in instance.answer_choices
                    )
                    labels.append(instance.correct_answer_index)
                tensors = tokenizer.batch_encode_plus(
                    texts,
                    padding=True,
                    truncation="only_first",
                    return_tensors="pt",
                    pad_to_multiple_of=8,
                )
                results = model(
                    return_dict=True,
                    **{key: tensor.view(len(batch), task.number_of_choices, -1) for key, tensor in tensors.items()})
                for instance, logits in zip(batch, results.logits.detach().cpu()):
                    yield MCTask.InstanceResult(instance.id, instance.correct_answer_index, logits)

    def do_qa(
        self,
        task: QATask,
        instances: Iterable[QATask.Instance],
        *,
        batch_size: int = 32
    ) -> Iterator[QATask.InstanceResult]:
        # TODO: Replace this with Huggingface pipeline
        model = AutoModelForQuestionAnswering.from_pretrained(self.pretrained_model_name_or_path).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        instances = Tqdm.tqdm(instances, desc="Processing instances")
        with torch.inference_mode():
            for batch in more_itertools.chunked(instances, batch_size):
                texts = [
                    ((instance.context or "").strip(), instance.question.strip())
                    for instance in batch
                ]
                # TODO: For questions that are too long, we should use striding.
                tensors = tokenizer.batch_encode_plus(
                    texts,
                    padding=True,
                    truncation="only_first",
                    return_tensors="pt",
                    pad_to_multiple_of=8,
                    return_offsets_mapping=True
                )
                offset_mapping = tensors["offset_mapping"]
                del tensors["offset_mapping"]
                results = model(return_dict=True, **tensors)
                answer_spans = get_best_spans(
                    results.start_logits.cpu().detach(),
                    results.end_logits.cpu().detach())
                for instance, span, token_offsets, text in zip(batch, answer_spans, offset_mapping, texts):
                    if not span.any():
                        answer = None
                    else:
                        start = token_offsets[span[0]][0]
                        end = token_offsets[span[1]][1]
                        answer = text[0][start:end]
                    yield QATask.InstanceResult(
                        instance,
                        instance.answer.strip(),
                        answer)
