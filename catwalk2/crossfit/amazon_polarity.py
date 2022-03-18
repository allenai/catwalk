import os
import datasets
import numpy as np

from catwalk2.crossfit.fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class AmazonPolarity(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "amazon_polarity"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "negative",
            1: "positive",
        }

    def get_train_test_lines(self, dataset):
        # only train set, manually split 20% data as test

        lines = self.map_hf_dataset_to_list(dataset, "train")

        np.random.seed(42)
        np.random.shuffle(lines)
        
        n = len(lines)

        train_lines = lines[:10000]
        test_lines = lines[10000:11000]

        return train_lines, test_lines

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        if split_name == "validation":
            split_name = "test" # hg datasets only has train/test
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append(("title: " + datapoint["title"] + " [SEP] content: " + datapoint["content"], self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('amazon_polarity')

def main():
    dataset = AmazonPolarity()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()