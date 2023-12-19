from catwalk.dependencies.lm_eval.base import Task, rf
from catwalk.dependencies.lm_eval.metrics import mean


class Eurlex(Task):
    DATASET_PATH = "multi_eurlex"
    # We will use only the English subset
    DATASET_NAME = "en"
    EUROVOC_CONCEPTS = [
        "social questions",
        "industry",
        "finance",
        "trade",
        "business and competition",
        "international relations",
        "agriculture, forestry and fisheries",
        "production, technology and research",
        "transport",
        "employment and working conditions",
        "politics",
        "law",
        "education and communications",
        "international organisations",
        "energy",
        "EUROPEAN UNION",
        "science",
        "agri-foodstuffs",
        "geography",
        "economics",
        "environment"
    ] 

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True
    
    def has_test_docs(self):
        return True
    
    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs
    
    def validation_docs(self):
        return self.dataset["validation"]
    
    def test_docs(self):
        return self.dataset["test"]
    
    def doc_to_text(self, doc):
        return "Output all concepts from the list of EuroVoc concepts that can be applied to the text that follows. EUROVOC CONCEPTS: " + ", ".join(self.EUROVOC_CONCEPTS) + "\nTEXT: " + doc["text"] + "\n" + "Concepts:"
    
    def should_decontaminate(self):
        return True
    
    def doc_to_decontamination_query(self, doc):
        return doc["text"]
    
    def doc_to_target(self, doc):
        target_concepts = [self.EUROVOC_CONCEPTS[i] for i in doc["labels"]]
        return ", ".join(target_concepts)
    
    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        conts = [rf.greedy_until(ctx, ["\n"])]
        return conts

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        target_concepts = [self.EUROVOC_CONCEPTS[i] for i in doc["labels"]]
        predicted_concepts = [result.strip() for result in results[0].split(",")]
        f1, precision, recall = self.get_metrics(predicted_concepts, target_concepts)
        return {"f1": f1, "precision": precision, "recall": recall}
    
    def get_metrics(self, predctions, targets):
        predictions_set = set(predctions)
        targets_set = set(targets)
        overlap = predictions_set.intersection(targets_set)
        precision = len(overlap) / len(predictions_set) if predictions_set else 0.0
        recall = len(overlap) / len(targets_set) if targets_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0
        return f1, precision, recall
    
    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {"f1": mean, "precision": mean, "recall": mean}
    
    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {"f1": True, "precision": True, "recall": True}