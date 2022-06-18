from catwalk.task import Task, InstanceFormat
from typing import Dict, Any, Optional, Sequence
from tango.common.sequences import MappedSequence
from datasets import load_dataset
import random
from tokenizers import ByteLevelBPETokenizer
from torchmetrics import SQuAD
from torchmetrics.functional.text.squad import _normalize_text



class MixedFewshotTask(Task):
    SQUADSHIFT_SPLITS = ['new_wiki', 'nyt', 'reddit', 'amazon']
    MRQA_TEST_SPLITS = [
                'BioASQ',
                'DROP',
                'DuoRC.ParaphraseRC',
                'RACE',
                'RelationExtraction',
                'TextbookQA'
            ]
    MRQA_VALIDATION_SPLITS = [
        'HotpotQA',
        'NaturalQuestionsShort',
        'NewsQA',
        'SQuAD',
        'SearchQA',
        'TriviaQA-web'
    ]
    def __init__(
        self,
        inference_dataset_name: str,
        *,
        random_seed: Optional[int] = None,
        version_override: Optional[str] = None
    ):
        super().__init__(version_override=version_override)
        assert inference_dataset_name in ['squadshifts', 'mrqa']
        self.inference_dataset_name = inference_dataset_name
        self.add_instance_conversion(InstanceFormat.MIXED_FEWSHOT_SQUADSHIFTS, self.instance_as_mixed_fewshot)
        self.few_shot_source = MappedSequence(lambda x: x, load_dataset('squad')['train'])
        assert all(len(ex['answers']['text']) > 0 for ex in self.few_shot_source)
        # We maintain our own random, to help with determinism.
        if random_seed is None:
            random_seed = 57885161 + 43112609    # What could be more perfect than the sum of two perfect numbers?
        self.random = random.Random(random_seed)
        self.add_metric("squad", self._squad_metric_wrapper)

    def has_split(self, split: str) -> bool:
        """Returns ``True`` if a split with the given name exists. ``False`` otherwise."""
        if self.inference_dataset_name == 'squadshifts':
            return split in self.SQUADSHIFT_SPLITS
        elif self.inference_dataset_name == 'mrqa':
            return split in (self.MRQA_TEST_SPLITS + self.MRQA_VALIDATION_SPLITS)
        else:
            raise NotImplementedError
            

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        """Returns the split with the given name."""
        assert self.has_split(split)
        if self.inference_dataset_name == 'squadshifts':
            ds = load_dataset(self.inference_dataset_name, split)
            assert list(ds.keys()) == ['test']
            ds = ds['test'] 
        elif self.inference_dataset_name == 'mrqa':
            ds = load_dataset(self.inference_dataset_name, split=('test' if split in self.MRQA_TEST_SPLITS else 'validation'))
            ds = ds.filter(lambda e: e['subset'] == split)
            ds = ds.rename_column('qid','id')
            def reformat_answers(e):
                e['answers'] = {'text':e['detected_answers']['text'], 'answer_start':[d['start'][0] for d in e['detected_answers']['char_spans']]}
                return e
            ds = ds.map(reformat_answers)      
        # HF datasets are not sequences, even though they sometimes pretend they are. So we apply this hack
        # to make them act like sequences.
        ds = MappedSequence(lambda x: x, ds)
        assert all(len(ex['answers']['text']) > 0 for ex in ds)
        return ds
    
    def instance_as_mixed_fewshot(self, instance: Dict[str, Any], num_shots: int = 0, tokenizer: Any = None):
        assert isinstance(tokenizer, ByteLevelBPETokenizer), "Other tokenizers not yet supported"
        if num_shots == 0:
            fewshot_examples_formatted = ""
        else:
            fewshot_examples_formatted = []
            sample = None
            while True:
                sample = self.random.sample(self.few_shot_source, num_shots)
                def is_sample_good(ex, q_max, a_max):
                    answer = ex['answers']['text'][0]
                    return (len(tokenizer.encode(ex['question']).ids) < q_max) and (len(tokenizer.encode(answer).ids) < a_max)
                if all(is_sample_good(ex, 50, 50) for ex in sample):
                    break
                print('skipping ICL sample beause answers or questions are too long') #TODO make this logging instead
            for ex in sample:
                answer = ex['answers']['text'][0]
                truncated_context = self._get_context_window(ex, ex['context'], answer, tokenizer=tokenizer)
                fewshot_examples_formatted.append(self._format_example(truncated_context, ex['question'], (' '+ answer)))
            fewshot_examples_formatted = '\n\n'.join(fewshot_examples_formatted)
        
        truncated_instance_context = self._get_context_window(instance, instance['context'],instance['answers']['text'][0], tokenizer, 200)
        instance_formated = self._format_example(truncated_instance_context, instance['question'])

        return fewshot_examples_formatted + '\n\n' + instance_formated
    def _format_example(self, truncated_context: str, question: str, answer: str = '') -> str:
        return f"Background: {truncated_context}\n\nQuestion: {question}\n\nAnswer:{answer}"

    def normalize_text(self, s):
            # required for MRQA because of noisy answers
            s = _normalize_text(s)
            return ''.join(s.split())
    
    def _get_context_window(self, instance: Dict[str, Any], full_context: str, answer: str, tokenizer: Any, window_size: int = 100, window_stride: int = 50):
        # window_size -= len(tokenizer.encode(self._format_example('', instance['question'],instance['answers']['text'][0])))
        answer_toks = tokenizer.encode(answer).ids
        assert len(answer_toks) < window_size
        answer = tokenizer.decode(tokenizer.encode(answer).ids)
        toks = tokenizer.encode(full_context).ids
        assert (self.normalize_text(answer) in self.normalize_text(tokenizer.decode(toks))) or (answer in tokenizer.decode(toks))
        for i in range(0, len(toks), window_stride):
            context_window = tokenizer.decode(toks[i:i+window_size])
            if (self.normalize_text(answer) in self.normalize_text(context_window)) or (answer in context_window):
                return context_window
        
        raise RuntimeError('Answer not found in context, this should not happen')

    def _squad_metric_wrapper(self, preds, target):
        metric = SQuAD()
        return metric(preds, target)