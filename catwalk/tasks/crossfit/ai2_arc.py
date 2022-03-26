import os
import datasets
import numpy as np

from catwalk.tasks.crossfit.fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class AI2_ARC(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "ai2_arc"
        self.task_type = "text to text"
        self.license = "unknown"

    def get_choices_and_answer_string(self, datapoint):
        answer_index = datapoint["answerKey"]
        choices_string = ""
        for i in range(len(datapoint["choices"]["label"])):
            if datapoint["choices"]["label"][i] == answer_index:
                answer_string = datapoint["choices"]["text"][i]
            choices_string += " (" + datapoint["choices"]["label"][i] + ") " + datapoint["choices"]["text"][i]
        return choices_string, answer_string

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            choices_string, answer_string = self.get_choices_and_answer_string(datapoint)
            lines.append((datapoint["question"] + choices_string, answer_string))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("ai2_arc", "ARC-Challenge")

def main():
    dataset = AI2_ARC()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()