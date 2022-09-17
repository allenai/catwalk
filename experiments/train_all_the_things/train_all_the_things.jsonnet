local debug = false;

local tasks = [
    #"arc_challenge",
    #"arc_easy",
    #"piqa",
    #"copa",
    #"sciq",
    "logiqa",
    #"hellaswag",
    #"openbookqa",
    "headqa_en",
    #"winogrande"
];

local models = [
    "bert-base-uncased",
    "bert-base-cased",
    "bert-large-uncased",
    "bert-large-cased",
    "roberta-base",
    "roberta-large",
    "deberta-v3-base",
    "deberta-v3-small",
    "deberta-v3-large",
    #"deberta-v2-xlarge",
    #"deberta-v2-xxlarge",
];

local random_seeds = if debug then [42, 1] else [
    42,
    1337,
    #2147483647,
    #1,
    1985
];


local effective_batch_size = if debug then 6 else 32;

local batch_size_for_model(model, task) =
    std.min(effective_batch_size,
        if debug then 3 else
        (
            if std.length(std.findSubstr("xxl", model)) > 0 then 2 else
            if std.length(std.findSubstr("xl", model)) > 0 then 4 else
            if std.length(std.findSubstr("large", model)) > 0 then
                (if std.length(std.findSubstr("deberta", model)) > 0 then 4 else 8) else
            effective_batch_size)
        ) / (
            if std.length(std.findSubstr("headqa", task)) > 0 || std.length(std.findSubstr("logiqa", task)) > 0 then 2 else 1
        );


local trained_model_step_name(task, model, seed) = "trained_model_" + task + "_" + model + "_" + seed;

local trained_model(task, model, seed) = {
    [trained_model_step_name(task, model, seed)]: {
        type: "catwalk::finetune",
        step_resources: { gpu_count: 1 },
        model: model,
        tasks: [task],
        random_seed: seed,
        batch_size: batch_size_for_model(model, task),
        grad_accum: effective_batch_size / self.batch_size,
        [if debug then "train_epochs"]: 3,
        wandb_entity: "allennlp",
        wandb_project: "catwalk"
    }
};


local predict_results_step_name(task, model, seed) = "instance_results_" + task + "_" + model + "_" + seed;

local predict_results(task, model, seed) = {
    [predict_results_step_name(task, model, seed)]: {
        type: "catwalk::predict",
        step_resources: { gpu_count: 1 },
        model: {type: "ref", ref: trained_model_step_name(task, model, seed)},
        task: task,
        batch_size: batch_size_for_model(model, task) * 2,
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
