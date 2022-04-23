local task_names = [
    "arc_challenge",
    "arc_easy",
    "boolq",
    "copa",
    "headqa",
    "hellaswag",
    "lambada",
    "logiqa",
    "mathqa",
    "mc_taco",
    "mrpc",
    "multirc",
    "openbookqa",
    "piqa",
    "piqa",
    "prost",
#    "pubmedqa",
    "qnli",
    "qqp",
    "race",
    "rte",
    "sciq",
    "sst",
    "triviaqa",
    "webqs",
    "wic",
    "winogrande",
    "wnli",
    "wsc"
];

local task_steps = std.foldl(
    function(x, task_name) x + {
        ["predict_" + task_name]: {
            type: "catwalk::predict",
            model: "eai::gpt2",
            task: task_name
        },
        ["calculate_" + task_name]: {
            type: "catwalk::calculate_metrics",
            model: "eai::gpt2",
            task: task_name,
            predictions: {ref: "predict_" + task_name}
        }
    },
    task_names,
    {}
);

{
    steps: task_steps + {
        print_results: {
            type: "print",
            input: std.map(
                function(task_name) {ref: "calculate_" + task_name},
                task_names
            )
        }
    }
}
