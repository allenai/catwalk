local debug = false;

local tasks = if debug then [
    // MC
    "arc_easy",
    "piqa",
    // Classification
    "rte",
    "mnli",
] else [
    // MC
    #"arc_challenge",
    "arc_easy",
    "piqa",
    #"copa",
    #"sciq",
    #"logiqa",
    #"hellaswag",
    #"openbookqa",
    #"headqa_en",
    #"winogrande",
    // Classification
    "rte",
    #"mnli",
    #"cola",
    "sst",
    #"qqp",
    #"qnli",
    #"mrpc"
];

# By default we validate every 100 batches, but some datasets are too large for that, so we have an override here.
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


local models2batchsize = if debug then {
    "rc::t5-small": 2,
    "rc::gpt2": 3
} else {
    #"rc::t5-small": 16,
    "rc::t5-base": 16,
    "rc::t5-large": 8,
    #"rc::t5-3b": 16,
    #"rc::t5-11b": 16,
    "rc::gpt2": 16,
    #"rc::gpt2-medium": 16,
    #"rc::gpt2-large": 16,
    "rc::gpt2-xl": 1,
    #"rc::bloom-560m": 16,
    #"rc::bloom-1b1": 16,
    #"rc::bloom-1b7": 16,
    #"rc::bloom-3b": 16,
    #"rc::bloom-7b1": 16,
    #"rc::opt-125m": 16,
    #"rc::opt-350m": 16,
    #"rc::opt-1.3b": 16,
    #"rc::opt-2.7b": 16,
    #"rc::opt-6.7b": 16,
    #"rc::opt-13b": 16,
    #"rc::opt-30b": 16,
    #"rc::opt-66b": 16,
    #"rc::gpt-j-6b": 16,
};

local models = std.objectFields(models2batchsize);

local batch_size_for_model(model) = models2batchsize[model];

local random_seeds = if debug then [42, 1] else [
    42,
    1337,
    #2147483647,
    #1,
    1985
];


local effective_batch_size = if debug then (2*3*4) else 16;

//
// Steps for full finetuning
//

local finetuned_model_step_name(task, model, seed) = "finetuned_model_" + task + "_" + model + "_" + seed;

local finetuned_model(task, model, seed) = {
    [finetuned_model_step_name(task, model, seed)]: {
        type: "catwalk::finetune",
        step_resources: { gpu_count: 1 },
        model: model,
        tasks: [task],
        random_seed: seed,
        batch_size: batch_size_for_model(model),
        grad_accum: effective_batch_size / self.batch_size,
        val_metric_name: "loss",
        validate_every: validate_every(task),
        [if debug then "train_steps"]: 3,
        [if debug then "train_epochs"]: null,
        [if debug then "validation_steps"]: 5,
        [if !debug then "wandb_entity"]: "allennlp",
        [if !debug then "wandb_project"]: "catwalk"
    }
};


local finetuned_results_step_name(task, model, seed) = "finetuned_results_" + task + "_" + model + "_" + seed;

local finetuned_results(task, model, seed) = {
    [finetuned_results_step_name(task, model, seed)]: {
        type: "catwalk::predict",
        step_resources: { gpu_count: 1 },
        model: {type: "ref", ref: finetuned_model_step_name(task, model, seed)},
        task: task,
        batch_size: batch_size_for_model(model) * 2,
        [if debug then "limit"]: 10
    }
};


local finetuned_metrics_step_name(task, model, seed) = "finetuned_metrics_" + task + "_" + model + "_" + seed;

local finetuned_metrics(task, model, seed) = {
    [finetuned_metrics_step_name(task, model, seed)]: {
        type: "catwalk::calculate_metrics",
        step_resources: { machine: "local" },
        model: {type: "ref", ref: finetuned_model_step_name(task, model, seed)},
        task: task,
        predictions: {type: "ref", ref: finetuned_results_step_name(task, model, seed)}
    }
};


//
// Steps for soft prompt tuning
//

local softprompt_model_step_name(task, model, seed) = "softprompt_model_" + task + "_" + model + "_" + seed;

local softprompt_model(task, model, seed) = {
    [softprompt_model_step_name(task, model, seed)]: {
        type: "catwalk::finetune",
        step_resources: { gpu_count: 1 },
        model: {
            "type": "catwalk::with_soft_prompt",
            "model": model,
            "prompt_length": 3,
            "random_seed": seed
        },
        tasks: [task],
        random_seed: seed,
        batch_size: batch_size_for_model(model),
        grad_accum: effective_batch_size / self.batch_size,
        val_metric_name: "loss",
        [if debug then "train_steps"]: 3,
        [if debug then "train_epochs"]: null,
        [if debug then "validation_steps"]: 5,
        [if !debug then "wandb_entity"]: "allennlp",
        [if !debug then "wandb_project"]: "catwalk"
    }
};


local softprompt_results_step_name(task, model, seed) = "softprompt_results_" + task + "_" + model + "_" + seed;

local softprompt_results(task, model, seed) = {
    [softprompt_results_step_name(task, model, seed)]: {
        type: "catwalk::predict",
        step_resources: { gpu_count: 1 },
        model: {type: "ref", ref: softprompt_model_step_name(task, model, seed)},
        task: task,
        batch_size: batch_size_for_model(model) * 2,
        [if debug then "limit"]: 10
    }
};


local softprompt_metrics_step_name(task, model, seed) = "softprompt_metrics_" + task + "_" + model + "_" + seed;

local softprompt_metrics(task, model, seed) = {
    [softprompt_metrics_step_name(task, model, seed)]: {
        type: "catwalk::calculate_metrics",
        step_resources: { machine: "local" },
        model: {type: "ref", ref: softprompt_model_step_name(task, model, seed)},
        task: task,
        predictions: {type: "ref", ref: softprompt_results_step_name(task, model, seed)}
    }
};



{
    steps: std.foldl(
        function(x, task) x + std.foldl(
            function(y, model) y + std.foldl(
                function(z, seed) z +
                    softprompt_model(task, model, seed) +
                    softprompt_results(task, model, seed) +
                    softprompt_metrics(task, model, seed) +
                    finetuned_model(task, model, seed) +
                    finetuned_results(task, model, seed) +
                    finetuned_metrics(task, model, seed),
                random_seeds,
                {}
            ),
            models,
            {}
        ),
        tasks,
        {}
    ) + {
        "tabulate": {
            type: "catwalk::tabulate_metrics",
            step_resources: { machine: "local" },
            metrics: std.foldl(
                function(x, task) x + std.foldl(
                    function(y, model) y + std.foldl(
                        function(z, seed) z + {
                            ["finetuning_" + task + "\t" + model + "\t" + seed]: {
                                type: "ref", ref: finetuned_metrics_step_name(task, model, seed)
                            },
                            ["softprompt_" + task + "\t" + model + "\t" + seed]: {
                                type: "ref", ref: softprompt_metrics_step_name(task, model, seed)
                            }
                        },
                        random_seeds,
                        {}
                    ),
                    models,
                    {}
                ),
                tasks,
                {}
            )
        }
    }
}
