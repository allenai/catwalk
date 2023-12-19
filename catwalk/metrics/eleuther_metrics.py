class EleutherMetrics:
    # Use directly metrics from Eleuther task
    def __init__(self, inner_task):
        self.inner_task = inner_task
        self.predictions = []

    def get_metrics(self, data):
        # Metrics are already calculated at instance level
        return data["metrics"]

    def update(self, data):
        self.predictions.append(data["metrics"])

    def compute(self):
        metrics = {
            key: fn([p[key] for p in self.predictions])
            for key, fn in self.inner_task.aggregation().items()
        }
        return metrics
