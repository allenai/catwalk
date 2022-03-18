import os
import datasets
import numpy as np

from catwalk2.crossfit.fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class PIQA(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "piqa"
        self.task_type = "classification"
        self.license = "unknown"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "solution 1",
            1: "solution 2"
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append(("goal: " + datapoint["goal"] + " [SEP] solution 1" + datapoint["sol1"] + " [SEP] solution 2" + datapoint["sol2"], datapoint["label"]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('piqa')

def main():
    dataset = PIQA()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()