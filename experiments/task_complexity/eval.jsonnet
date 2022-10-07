local debug = false;

local tasks = [
    "arc_challenge",
    "arc_easy",
    "copa",
    "piqa",
    "boolq",
    "hellaswag",
    "openbookqa",
    "rte",
    "winogrande",
    "wic",
    "sst"
];

local models2batchsize = if debug then {
    "gpt2": 3,
    "tiny-gpt2": 4
} else {
    "gpt2": 32,
    "gpt2-medium": 32,
    "gpt2-large": 16,
    "gpt2-xl": 8,
    "bloom-560m": 16,
    "bloom-1b1": 16,
    "bloom-1b7": 16,
    "bloom-3b": 4,
    "bloom-7b1": 4,
    "opt-125m": 64,
    "opt-350m": 64,
    "opt-1.3b": 16,
    "opt-2.7b": 8,
    "opt-6.7b": 4,
    "opt-30b": 32,
    "opt-66b": 32,
    "gpt-j-6b": 4,
    "gpt-neo-125m": 64,
    "gpt-neo-1.3b": 16,
    "gpt-neo-2.7b": 8,
    "gpt-neox-20b": 32,
};

local models = std.objectFields(models2batchsize);

local batch_size_for_model(model) = models2batchsize[model];

local gpus_for_model_dict = {
    "opt-30b": 2,
    "opt-66b": 4,
    "gpt-neox-20b": 2,
};

local gpus_for_model(model) = if std.objectHas(gpus_for_model_dict, model) then gpus_for_model_dict[model] else 1;


local predict_results_step_name(task, model) = "instance_results_" + task + "_" + model;

local predict_results(task, model) = {
    [predict_results_step_name(task, model)]: {
        type: "catwalk::predict",
        step_resources: { gpu_count: gpus_for_model(model) },
        model: "promptsource::" + model,
        task: task,
        batch_size: batch_size_for_model(model),
        limit: if debug then 10 else 1000
    }
};


local metrics_results_step_name(task, model) = "metrics_" + task + "_" + model;

local metrics_results(task, model) = {
    [metrics_results_step_name(task, model)]: {
        type: "catwalk::calculate_metrics",
        model: "promptsource::" + model,
        task: task,
        predictions: {type: "ref", ref: predict_results_step_name(task, model)}
    }
};


{
    steps: std.foldl(
        function(x, task) x + std.foldl(
            function(y, model) y +
                predict_results(task, model) +
                metrics_results(task, model),
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
                    function(y, model) y + {
                        [task + "\t" + model]: {type: "ref", ref: metrics_results_step_name(task, model)}
                    },
                    models,
                    {}
                ),
                tasks,
                {}
            ),
        }
    }
}
