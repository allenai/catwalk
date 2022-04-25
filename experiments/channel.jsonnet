local task_names = [
    "arc_challenge",
    "arc_easy",
#    "boolq",
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
#    "race",
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

local tasks_without_validation = [
    "prost",
    "webqs"
];

local task_steps(prefix, model_name) = std.foldl(
    function(x, task_name) x + {
        [prefix + "_predict_" + task_name]: {
            type: "catwalk::predict",
            model: model_name,
            task: task_name,
            split: if std.member(tasks_without_validation, task_name) then "test" else "validation",
            batch_size: 64
        },
        [prefix + "_calculate_" + task_name]: {
            type: "catwalk::calculate_metrics",
            model: "eai::gpt2",
            task: task_name,
            predictions: {ref: prefix + "_predict_" + task_name}
        }
    },
    task_names,
    {}
);

{
    steps: task_steps("direct", "eai::gpt2") + {
        direct_print_results: {
            type: "print",
            input: std.foldl(
                function(x, task_name) x + { [task_name]: {ref: "direct_calculate_" + task_name} },
                task_names,
                {}
            )
        }
    } + task_steps("channel", "eai::channel_gpt2") + {
        channel_print_results: {
            type: "print",
            input: std.foldl(
                function(x, task_name) x + { [task_name]: {ref: "channel_calculate_" + task_name} },
                task_names,
                {}
            )
        }
    }
}
