local debug = false;

local tasks = [
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
    "mnli_mismatched",
    "cola",
    "sst",
    "qqp",
    "qnli",
    "mrpc"
];

local models2batchsize = if debug then {
    "bert-base-uncased": 2,
    "roberta-base": 3,
    "deberta-v3-base": 4,
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
    "deberta-v2-xlarge": 2,
    "deberta-v2-xxlarge": 1,
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

local trained_model_step_name(task, model, seed) = "trained_model_" + task + "_" + model + "_" + seed;

local trained_model(task, model, seed) = {
    [trained_model_step_name(task, model, seed)]: {
        type: "catwalk::finetune",
        step_resources: { gpu_count: 1 },
        model: model,
        tasks: [task],
        random_seed: seed,
        batch_size: batch_size_for_model(model),
        grad_accum: effective_batch_size / self.batch_size,
        val_metric_name: "acc",
        [if debug then "train_steps"]: 3,
        [if debug then "train_epochs"]: null,
        [if debug then "validation_steps"]: 5,
        [if !debug then "wandb_entity"]: "allennlp",
        [if !debug then "wandb_project"]: "catwalk"
    }
};


local predict_results_step_name(task, model, seed) = "instance_results_" + task + "_" + model + "_" + seed;

local predict_results(task, model, seed) = {
    [predict_results_step_name(task, model, seed)]: {
        type: "catwalk::predict",
        step_resources: { gpu_count: 1 },
        model: {type: "ref", ref: trained_model_step_name(task, model, seed)},
        task: task,
        batch_size: batch_size_for_model(model) * 2,
        [if debug then "limit"]: 10
    }
};


local metrics_results_step_name(task, model, seed) = "metrics_" + task + "_" + model + "_" + seed;

local metrics_results(task, model, seed) = {
    [metrics_results_step_name(task, model, seed)]: {
        type: "catwalk::calculate_metrics",
        model: {type: "ref", ref: trained_model_step_name(task, model, seed)},
        task: task,
        predictions: {type: "ref", ref: predict_results_step_name(task, model, seed)}
    }
};


{
    steps: std.foldl(
        function(x, task) x + std.foldl(
            function(y, model) y + std.foldl(
                function(z, seed) z +
                    trained_model(task, model, seed) +
                    predict_results(task, model, seed) +
                    metrics_results(task, model, seed),
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
            metrics: std.foldl(
                function(x, task) x + std.foldl(
                    function(y, model) y + std.foldl(
                        function(z, seed) z + {
                            [task + "\t" + model + "\t" + seed]: {type: "ref", ref: metrics_results_step_name(task, model, seed)}
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
