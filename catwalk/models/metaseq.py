import collections
from typing import Sequence, Dict, Any, Iterator, Callable, Mapping, List, Tuple, Protocol

import more_itertools
import torch
from lm_eval.base import Request
from tango.common import Tqdm
from tango.integrations.torch.util import resolve_device
from torch import log_softmax
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, \
    AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5TokenizerFast

from catwalk.task import Task, InstanceFormat
from catwalk.model import Model
from catwalk.tasks.eleuther import EleutherTask

#### metaseq imports
import os
import random
import sys
import logging
import signal

import torch

from metaseq import options
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as dist_utils
from metaseq.hub_utils import GeneratorInterface
from metaseq.service.utils import encode_fn, build_logger

# mutable global hacks
# inputs = None
# outputs = None

@Model.register("metaseq::opt")
class MetaseqOPT(Model):
    MAX_SEQ_LEN = 2048
    # BATCH_SIZE = 2048  # silly high bc we dynamically batch by MAX_BATCH_TOKENS
    # MAX_BATCH_TOKENS = 3072
    BATCH_SIZE = 2
    DEFAULT_PORT = 6010
    MODEL_PARALLEL = 8
    TOTAL_WORLD_SIZE = 8

    # MODEL_SHARED_FOLDER should point to a shared drive (e.g. NFS) where the
    # checkpoints from S3 are stored. As an example:
    # MODEL_SHARED_FOLDER = "/example"
    # $ ls /example
    # dict.txt  gpt2-merges.txt  gpt2-vocab.json  OPT-175B
    # SPECIFIC_MODEL_FOLDER = "OPT-175B"
    # $ ls /example/OPT-175B/reshard_no_os
    # reshard-model_part-0.pt
    # reshard-model_part-1.pt
    # reshard-model_part-2.pt
    # reshard-model_part-3.pt
    # reshard-model_part-4.pt
    # reshard-model_part-5.pt
    # reshard-model_part-6.pt
    # reshard-model_part-7.pt
    MODEL_SHARED_FOLDER = "/net/nfs.cirrascale/s2-research/opt-175b/checkpoints/"
    # LOCAL_SSD is optional, but it's assuming you have some sort of local
    # hard disk where we can cache a copy of the weights for faster loading.
    LOCAL_SSD = ""
    if not LOCAL_SSD:
        # don't use local cache
        LOCAL_SSD = MODEL_SHARED_FOLDER

    CHECKPOINT_FOLDER = MODEL_SHARED_FOLDER #os.path.join(MODEL_SHARED_FOLDER, "OPT-175B", "reshard_no_os")
    CHECKPOINT_LOCAL = os.path.join("/net/nfs.cirrascale/s2-research/opt-175b/checkpoints/", "reshard.pt")

    # tokenizer files
    BPE_MERGES = os.path.join(MODEL_SHARED_FOLDER, "gpt2-merges.txt")
    BPE_VOCAB = os.path.join(MODEL_SHARED_FOLDER, "gpt2-vocab.json")

    LAUNCH_ARGS = [
        f"--model-parallel-size {MODEL_PARALLEL}",
        f"--distributed-world-size {TOTAL_WORLD_SIZE}",
        "--task language_modeling",
        f"--bpe-merges {BPE_MERGES}",
        f"--bpe-vocab {BPE_VOCAB}",
        "--bpe hf_byte_bpe",
        f"--merges-filename {BPE_MERGES}",
        f"--vocab-filename {BPE_VOCAB}",
        f"--path {CHECKPOINT_LOCAL}",
        "--beam 1 --nbest 1",
        "--distributed-port 13000",
        "--checkpoint-shard-count 1",
        "--use-sharded-state",
        f"--batch-size {BATCH_SIZE}",
        f"--buffer-size {BATCH_SIZE * MAX_SEQ_LEN}",
        f"--max-tokens {BATCH_SIZE * MAX_SEQ_LEN}",
        "/tmp",  # required "data" argument.
    ]
    def __init__(self):
        # TODO specify model size here?
        pass

    def predict(  # type: ignore
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        *args,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        instance_index_to_request_indices: Mapping[int, Mapping[str, List[int]]] = \
            collections.defaultdict(lambda: collections.defaultdict(list))
        requests: Mapping[str, List[Request]] = collections.defaultdict(list)

        # get all the requests
        for instance_index, instance in enumerate(instances):
            instance_requests = task.convert_instance(instance, InstanceFormat.ELEUTHER_REQUESTS)
            if not isinstance(instance_requests, (list, tuple)):
                instance_requests = [instance_requests]
            for instance_request in instance_requests:
                request_type = instance_request.request_type
                instance_index_to_request_indices[instance_index][request_type].append(len(requests[request_type]))
                requests[request_type].append(instance_request)

        # run the requests
        results: Dict[str, Sequence] = {}
        class InferenceFunc(Protocol):
            def __call__(self, requests: Sequence[Request], **kwargs) -> Sequence: ...
        request_type_to_fn: Mapping[str, InferenceFunc] = {
            "loglikelihood": self._run_loglikelihood,
            "loglikelihood_rolling": self._run_loglikelihood_rolling,
            "greedy_until": self._run_greedy_until
        }
        for request_type, requests_per_type in requests.items():
            results[request_type] = request_type_to_fn[request_type](
                requests_per_type,
                **kwargs
            )
        assert isinstance(task, EleutherTask), "We can only calculate metrics for EleutherTasks."
        for instance_index, instance in enumerate(instances):
            doc = task.convert_instance(instance, InstanceFormat.ELEUTHER_DOC)

            results_for_instance: List = []
            for request_type, request_indices in instance_index_to_request_indices[instance_index].items():
                results_for_instance.extend(results[request_type][i] for i in request_indices)

            yield task.inner_task.process_results(doc, results_for_instance)

    def _run_loglikelihood(
        self,
        requests: Sequence[Request],
        **kwargs
    ) -> Sequence:
        # temporaray bypass of reference scoring functionality
        return [(float('-inf'), False)] * len(requests)

    def _run_loglikelihood_rolling(
        self,
        requests: Sequence[Request],
        **kwargs
    ) -> Sequence:
        raise NotImplementedError

    def _run_greedy_until(
        self,
        requests: Sequence[Request],
        **kwargs
    ) -> Sequence:
        output = self.generate([r.args[0] for r in requests])
        return output

    def calculate_metrics(self, task: Task, predictions: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        assert isinstance(task, EleutherTask), "We can only calculate metrics for EleutherTasks."
        return {
            key: fn([p[key] for p in predictions])
            for key, fn in task.inner_task.aggregation().items()
        }
    
    def generate(self, prompts: Sequence[str]) -> Sequence[str]:
        return self._init_model_and_gen(prompts)
    
    def _init_model_and_gen(self, prompts: Sequence[str]) -> Sequence[str]:
        # parse args
        parser = options.get_generation_parser()
        # dumb defaults overriding
        parser.set_defaults(lr_scheduler=None, criterion=None)
        flat_launch_args = []
        for s in self.LAUNCH_ARGS:
            flat_launch_args += s.split()
        args = options.parse_args_and_arch(parser, input_args=flat_launch_args)
        args.data = os.path.dirname(args.path)  # hardcode the data arg
        cfg = convert_namespace_to_omegaconf(args)
        cfg.distributed_training.distributed_world_size = self.TOTAL_WORLD_SIZE

        # spawn distributed
        dist_utils.infer_init_method(cfg.distributed_training)
        start_rank = cfg.distributed_training.distributed_rank
        cfg.distributed_training.distributed_rank = None  # assign automatically
        kwargs = {
            "start_rank": start_rank,
            "namespace_args": args
        }
        spawncontext = torch.multiprocessing.start_processes(
            self._distributed_main,
            # need to give rank offset as 1 to cover the fact that the main
            # process is rank 0, but that spawn() doesn't let you control rank:
            # it always starts at 0
            (prompts, self._worker_main, cfg, kwargs),
            nprocs=min(
                torch.cuda.device_count(),
                cfg.distributed_training.distributed_world_size - 1,
            ),
            join=False,
            start_method="spawn",
        )
        try:
            # -1 because we offset by +1 inside distributed_main when using
            # spawn_helper
            retval = self._distributed_main(-1, prompts, self._worker_main, cfg, kwargs)
            spawncontext.join()
            return retval
        except (KeyboardInterrupt, Exception):
            # weirdly KeyboardInterrupt is not an Exception
            # propagate exceptions on the main node by killing workers
            for p in spawncontext.processes:
                if p.is_alive():
                    os.kill(p.pid, signal.SIGTERM)
            raise

    def _distributed_main(self, i, prompts: Sequence[str], main, cfg: MetaseqConfig, kwargs) -> Sequence[str]:
        if not cfg.distributed_training.distributed_no_spawn:
            # if in local spawning, i is offset by -1 since torch.multiprocessing.spawn
            # always starts at rank 0
            i = i + 1
        cfg.distributed_training.device_id = i
        if torch.cuda.is_available() and not cfg.common.cpu:
            torch.cuda.set_device(cfg.distributed_training.device_id)
            # This is temporary way of making microsoft Tutel happy, as it reads the local rank from
            # the env. To make it work in cleaner way, we might need to change their interfaces to be
            # able to pass local rank.
            os.environ["LOCAL_RANK"] = str(cfg.distributed_training.device_id)
        if cfg.distributed_training.distributed_rank is None:
            # start_rank is the rank of gpu 0 on this machine.
            cfg.distributed_training.distributed_rank = kwargs.pop("start_rank", 0) + i

        cfg.distributed_training.distributed_rank = dist_utils.distributed_init(cfg)

        after_distributed_init_fn = kwargs.pop("after_distributed_init_fn", None)
        if after_distributed_init_fn:
            cfg = after_distributed_init_fn(cfg)
        output = main(prompts, cfg, **kwargs)

        if torch.distributed.is_initialized():
            torch.distributed.barrier(dist_utils.get_global_group())

        return output

    def _worker_main(self, prompts: Sequence[str], cfg: MetaseqConfig, **kwargs) -> Sequence[str]:
        # make sure generations are stochastic since we have many workers TODO understand this
        torch.manual_seed(random.randint(1, 20000))
        torch.cuda.manual_seed(random.randint(1, 20000))

        generator = GeneratorInterface(cfg)
        models = generator.load_model()  # noqa: F841

        # logger.info(f"loaded model {cfg.distributed_training.distributed_rank}") TODO delete?
        request_object = dist_utils.broadcast_object(
            None, src_rank=0, group=dist_utils.get_global_group()
        )
        if torch.distributed.get_rank() == 0:
            # stuff for main proccess
            return self._rank0_worker_main(prompts, generator)
        else:
            for i in range(len(prompts)):
                # useful in FSDP setting
                request_object = dist_utils.broadcast_object(
                    None, src_rank=0, group=dist_utils.get_global_group()
                )
                _ = generator.generate(**request_object)
            return _

    def _rank0_worker_main(self, prompts: Sequence[str], generator) -> Sequence[str]:
        """
        TODO
        """
        def tokenize_strings(generator, strings):
            return [encode_fn(generator, s) for s in strings]
        outputs = []
        for i in Tqdm.tqdm(list(range(0,len(prompts),self.BATCH_SIZE)), desc="running generation inference"):
            tokenized_prompts = tokenize_strings(generator, prompts[i:i+self.BATCH_SIZE])

            request_object = {
                'inputs' : tokenized_prompts,
                'max_tokens': [256] * len(tokenized_prompts),
                'temperature': 0.7,
                'top_p': 0.9,
                'n': 1
            }
            dist_utils.broadcast_object(
                            request_object, src_rank=0, group=dist_utils.get_global_group()
                        )
            generations = generator.generate(**request_object)
            output = [doc[0]['text'] for doc in generations]
            outputs.extend(output)

        return outputs