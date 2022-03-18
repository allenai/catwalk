import os
import datasets
import numpy as np

from catwalk.tasks.crossfit.fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class Emo(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "emo"
        self.task_type = "classification"
        self.license = "unknown"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0:"others",
            1:"happy",
            2:"sad",
            3:"angry",
        }
    
    def get_train_test_lines(self, dataset):
        train_lines = self.map_hf_dataset_to_list(dataset, "train")
        test_lines = self.map_hf_dataset_to_list(dataset, "test")

        return train_lines, test_lines

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append((datapoint["text"].strip(), self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('emo')

def main():
    dataset = Emo()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()