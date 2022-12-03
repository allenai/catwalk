# This experiment calculates the big matrix of model/method X tasks, with an accuracy score for each.

local debug = true;


#
# Define tasks
#

local tasks = if debug then [
        "arc_easy",
        "rte",
    ] else [
        // MC
        "arc_challenge",
        "arc_easy",
        "piqa",
        "copa",
        "sciq",
        "logiqa",
        "hellaswag",
        "openbookqa",
        "headqa_en",
        "winogrande",
        // Classification
        "rte",
        "mnli",
        //"mnli_mismatched", // Broken because it doesn't have a "validation" set.
        "cola",
        "sst",
        "qqp",
        "qnli",
        "mrpc"
    ];

local task2validate_every = {
    "qqp": 1000,
    "mnli": 1000,
    "mnli_mismatched": 1000,
    "sst": 1000,
    "hellaswag": 1000,
    "winogrande": 1000,
    "logicqa": 200,
    "sciq": 500
};

local validate_every(task) = std.get(task2validate_every, task, 100);


#
# Define models
#

local trainable_models2batchsize = if debug then {
    "bert-base-uncased": 2,
    "roberta-base": 3,
    "deberta-v3-base": 5,
} else {
    "bert-base-uncased": 16,
    "bert-base-cased": 16,
    "bert-large-uncased": 16,
    "bert-large-cased": 16,
    "roberta-base": 16,
    "roberta-large": 16,
    "deberta-v3-base": 16,
    "deberta-v3-small": 16,
    "deberta-v3-large": 8,
    "deberta-v2-xlarge": 4,
    "deberta-v2-xxlarge": 2,
};

local shot_models2batchsize = if debug then {
    "rc::t5-small": 2,
    "rc::gpt2": 3,
    "rc::bloom-560m": 5,
    "rc::opt-125m": 7,
} else {
    "rc::t5-small": 16,
    "rc::t5-base": 16,
    "rc::t5-large": 16,
    "rc::t5-3b": 16,
    "rc::t5-11b": 16,
    "rc::gpt2": 16,
    "rc::gpt2-medium": 16,
    "rc::gpt2-large": 16,
    "rc::gpt2-xl": 16,
    "rc::bloom-560m": 16,
    "rc::bloom-1b1": 16,
    "rc::bloom-1b7": 16,
    "rc::bloom-3b": 16,
    "rc::bloom-7b1": 16,
    "rc::opt-125m": 16,
    "rc::opt-350m": 16,
    "rc::opt-1.3b": 16,
    "rc::opt-2.7b": 16,
    "rc::opt-6.7b": 16,
    #"rc::opt-13b": 16,
    #"rc::opt-30b": 16,
    #"rc::opt-66b": 16,
    "rc::gpt-j-6b": 16,
};

local shot_models = std.objectFields(shot_models2batchsize);
local trainable_models = std.objectFields(trainable_models2batchsize);

local batch_size_for_model(model) = std.get(
    shot_models2batchsize,
    model,
    std.get(trainable_models2batchsize, model));


local random_seeds = if debug then [42, 1337] else [
    42,
    1337,
    #2147483647,
    #1,
    1985
];


#
# Cross products
#

local training_configs = std.flatMap(
    function(t) std.flatMap(
        function(m) std.map(
            function(s) { "task": t, "model": m, "seed": s },
            random_seeds
        ),
        trainable_models
    ),
    tasks
);

local non_training_configs = std.flatMap(
    function(t) std.map(
        function(m) { "task": t, "model": m, "seed": null },
        shot_models
    ),
    tasks
);

local all_configs = training_configs + non_training_configs;


#
# Training steps
#

local effective_batch_size = if debug then 2*3*5*7 else 16;

local trained_model_step_name(config) = "trained_model_" + config.task + "_" + config.model + "_" + config.seed;

local trained_models = std.foldl(
    function(x, config) x + {
        [trained_model_step_name(config)]: {
            type: "catwalk::finetune",
            step_resources: { gpu_count: 1 },
            model: config.model,
            tasks: [config.task],
            random_seed: config.seed,
            batch_size: batch_size_for_model(config.model),
            grad_accum: effective_batch_size / self.batch_size,
            val_metric_name: "acc",
            validate_every: validate_every(config.task),
            [if debug then "train_steps"]: 3,
            [if debug then "train_epochs"]: null,
            [if debug then "validation_steps"]: 5,
            [if !debug then "wandb_entity"]: "allennlp",
            [if !debug then "wandb_project"]: "catwalk",
            early_stopping_patience: 6
        }
    },
    training_configs,
    {}
);

local model_ref(config) =
    if std.count(training_configs, config) > 0
    then {type: "ref", ref: trained_model_step_name(config)}
    else config.model;


#
# Prediction steps
#

local predictions_step_name(config) =
    if std.count(training_configs, config) > 0
    then "instance_results_" + config.task + "_" + config.model + "_" + config.seed
    else "instance_results_" + config.task + "_" + config.model;

local predictions = std.foldl(
    function(x, config) x + {
        [predictions_step_name(config)]: {
            type: "catwalk::predict",
            step_resources: { gpu_count: 1 },
            model: model_ref(config),
            task: config.task,
            batch_size: batch_size_for_model(config.model),
            [if debug then "limit"]: 10
        }
    },
    all_configs,
    {}
);


#
# Metrics steps
#

local metrics_step_name(config) =
    if std.count(training_configs, config) > 0
    then "metrics_" + config.task + "_" + config.model + "_" + config.seed
    else "metrics_" + config.task + "_" + config.model;

local metrics = std.foldl(
    function(x, config) x + {
        [metrics_step_name(config)]: {
            type: "catwalk::calculate_metrics",
            model: model_ref(config),
            task: config.task,
            predictions: {type: "ref", ref: predictions_step_name(config)}
        }
    },
    all_configs,
    {}
);


#
# Putting it all together
#

{
    "steps":
        trained_models +
        predictions +
        metrics + {
            "tabulate": {
                type: "catwalk::tabulate_metrics",
                metrics: std.foldl(
                    function(x, config) x + {
                        [
                            if std.count(training_configs, config) > 0
                            then config.task + "_" + config.model + "_" + config.seed
                            else config.task + "_" + config.model
                        ]: {type: "ref", ref: metrics_step_name(config)}
                    },
                    all_configs,
                    {}
                ),
            }
        }
}
