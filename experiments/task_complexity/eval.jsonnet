local debug = false;

local tasks = [
    "arc_challenge",
    "arc_easy",
    "boolq",
    "copa",
    "hellaswag",
    "openbookqa",
    "piqa",
    "rte",
    "sst",
    "wic",
    "winogrande",
    "lambada",
    #"pile",   # TODO, only perplexity task
    #"logiqa", # Doesn't have promptsource.
    "mc_taco",
    "mrpc",
    "multirc",
    #"prost",    # Prost is zero-shot only, and doesn't have promptsource.
    "pubmedqa",
    "qnli",
    "qqp",
    "sciq",
    #"triviaqa",    # Open-ended QA
    #"webqs",       # Open-ended QA
    "wnli",
    "wsc",
    "race",
    "headqa_en",
    "mathqa",
    # Arithmetic is not in promptsource, and is open-ended QA.
    #"arithmetic_2da",
    #"arithmetic_2ds",
    #"arithmetic_3da",
    #"arithmetic_3ds",
    #"arithmetic_4da",
    #"arithmetic_4ds",
    #"arithmetic_5da",
    #"arithmetic_5ds",
    #"arithmetic_2dm",
    #"arithmetic_1dc",
];

local models2batchsize = if debug then {
    "gpt2": 2,
    "tiny-gpt2": 3,
    "t5-very-small-random": 4
} else {
    // decoder-only
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
    "opt-30b": 16,
    "opt-66b": 8,
    "gpt-j-6b": 4,
    "gpt-neo-125m": 64,
    "gpt-neo-1.3b": 16,
    "gpt-neo-2.7b": 8,
    "gpt-neox-20b": 16,
    // encoder/decoder
    "t5-small-lm-adapt": 64,
    "t5-base-lm-adapt": 32,
    "t5-large-lm-adapt": 8,
    "t5-xl-lm-adapt": 2,
    "t5-xxl-lm-adapt": 1,
    "t5-v1_1-small": 64,
    "t5-v1_1-base": 32,
    "t5-v1_1-large": 8,
    "t5-v1_1-xl": 2,
    "t5-v1_1-xxl": 1,
    #"t0": 1,
    #"t0p": 1,
    #"t0pp": 1,
    #"t0_single_prompt": 1,
    #"t0_original_task_only": 1,
    #"t0-3b": 8,
    #"ct0-11b": 1
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
