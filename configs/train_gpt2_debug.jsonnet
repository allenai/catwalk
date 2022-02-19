# This file contains some settings that are small enough so I can debug on my laptop.

local train = import 'train.libsonnet';
train(pretrained_model="gpt2", training_steps=4, validation_steps=3, batch_size=3)