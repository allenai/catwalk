import argparse
import os
from tango.common.logging import initialize_logging
import time

from catwalk.models import MetaICLModel
from catwalk.steps import CalculateMetricsStep, PredictStep
from catwalk.tasks import TASK_SETS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zeroshot', action='store_true')
    parser.add_argument('--no_prefix_caching', action='store_true')
    parser.add_argument('--first_n_tasks', type=int, default=20)
    args = parser.parse_args()

    start = time.time()
    initialize_logging(log_level="ERROR")
    os.environ['TOKENIZERS_PARALLELISM'] = "false"

    tasks = TASK_SETS['metaicl-classification-eval']
    tasks = sorted(tasks)[:args.first_n_tasks]

    num_shots = 0 if args.zeroshot else 16
    if args.zeroshot:
        batch_size = 64    
    elif args.no_prefix_caching: 
        batch_size = 16 # to account for larger input sizes with ICL
    # CACHING with batching does not work close to the max model size as
    # the largest prefix + largest continuation in a batch must be <= max model size
    else:
        batch_size = 1
    limit = 1000
    random_subsample_seed=42
    seeds = [100] if args.zeroshot else [100, 13, 21, 42, 87]

    model = MetaICLModel('gpt2-large', continuation_seperator = ' ' if args.zeroshot else  '\n', prefix_caching = not args.no_prefix_caching) 

    seed2metrics = {}
    for fewshot_seed in seeds:
        metric_task_dict = {}
        for task in tasks:

            predictions = PredictStep(
                model=model,
                task=task,
                batch_size=batch_size,
                limit=limit,
                random_subsample_seed=random_subsample_seed,
                num_shots=num_shots,
                fewshot_seed=fewshot_seed,
                )
            metrics = CalculateMetricsStep(
                model=model,
                task=task,
                predictions=predictions)
            metric_task_dict[task] = metrics
        seed2metrics[fewshot_seed] = metric_task_dict

    avg_f1_per_seed = []
    avg_acc_per_seed = []
    for seed, metric_task_dict in seed2metrics.items():
        total_sum_f1 = 0.0
        total_sum_acc = 0.0
        for task, metrics in metric_task_dict.items():
            for metric, result in metrics.result().items():
                avg_result = result.mean()
                if metric == 'f1':
                    total_sum_f1 += avg_result.item()
                elif metric == 'acc':
                    total_sum_acc += avg_result.item()
                print(f"{task}\t{seed}\t{metric}\t{avg_result}")
        avg_f1_per_seed.append(total_sum_f1 / len(tasks))
        avg_acc_per_seed.append(total_sum_acc / len(tasks))
    
    print(f"avg macro f1 over seeds {sum(avg_f1_per_seed) / len(seeds)}")
    print(f"min macro f1 over seeds {min(avg_f1_per_seed)}")
    print(f"avg macro acc over seeds {sum(avg_acc_per_seed) / len(seeds)}")
    print(f"min macro acc over seeds {min(avg_acc_per_seed)}")
    
    end = time.time()
    print(f"total seconds elapsed: {end - start}")

if __name__ == "__main__":

    main()