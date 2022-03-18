import os
import datasets
import numpy as np

from catwalk2.crossfit.fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class ANLI(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "anli"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "entailment",
            1: "neutral",
            2: "contradiction",
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        if split_name == "validation":
            split_name = "dev"
        split_name = split_name + "_r1"
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append(("premise: " + datapoint["premise"].replace("\n", " ") + " [SEP] hypothesis: " + datapoint["hypothesis"].replace("\n", " "), self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('anli')

def main():
    dataset = ANLI()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()