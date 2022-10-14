local debug = false;

local tasks = [
    "arc_challenge",
    "arc_easy",
    "piqa",
    "copa",
    "sciq",
    "logiqa",
    "hellaswag",
    "openbookqa",
    "headqa_en",
    "winogrande"
];

local models = if debug then [
    "rc::t5-small",
    "rc::gpt2",
    "rc::bloom-560m",
    "rc::opt-125m",
] else [
    "rc::t5-small",
    "rc::t5-base",
    "rc::t5-large",
    "rc::t5-3b",
    "rc::t5-11b",
    "rc::gpt2",
    "rc::gpt2-medium",
    "rc::gpt2-large",
    "rc::gpt2-xl",
    "rc::bloom-560m",
    "rc::bloom-1b1",
    "rc::bloom-1b7",
    "rc::bloom-3b",
    "rc::bloom-7b1",
    "rc::opt-125m",
    "rc::opt-350m",
    "rc::opt-1.3b",
    "rc::opt-2.7b",
    "rc::opt-6.7b",
    #"rc::opt-13b",
    #"rc::opt-30b",
    #"rc::opt-66b",
    "rc::gpt-j-6b",
];


local batch_size_for_model(model, task) = if debug then 3 else 16;


local predict_results_step_name(task, model) = "instance_results_" + task + "_" + model;

local predict_results(task, model) = {
    [predict_results_step_name(task, model)]: {
        type: "catwalk::predict",
        step_resources: { gpu_count: 1 },
        model: model,
        task: task,
        batch_size: batch_size_for_model(model, task),
        [if debug then "limit"]: 10
    }
};


local metrics_results_step_name(task, model) = "metrics_" + task + "_" + model;

local metrics_results(task, model) = {
    [metrics_results_step_name(task, model)]: {
        type: "catwalk::calculate_metrics",
        model: model,
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
