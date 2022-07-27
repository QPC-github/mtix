import math
from unittest import TestCase


EPS = 1e-9


def compute_metrics(y_true, y_pred, qui_filter=None):

    def _extract_triples(results):
        triple_set = set()
        for citation_prediction in results:
            pmid = citation_prediction["PMID"]
            for descriptor_prediction in citation_prediction["Indexing"]:
                dui = descriptor_prediction["ID"]
                if "Subheadings" in descriptor_prediction:
                    for subheading_prediction in descriptor_prediction["Subheadings"]:
                        qui = subheading_prediction["ID"]
                        if qui_filter is None or qui in qui_filter:
                            triple_set.add((pmid, dui, qui))
                else:
                    triple_set.add((pmid, dui, ""))
        return triple_set

    pred_pmids = {e["PMID"] for e in y_pred}
    y_true = [e for e in y_true if e["PMID"] in pred_pmids]

    true_triples = _extract_triples(y_true)
    pred_triples = _extract_triples(y_pred)

    true_count = len(true_triples)
    pred_count = len(pred_triples)
    match_count = len(true_triples.intersection(pred_triples))

    p = match_count / (pred_count + EPS)
    r = match_count / (true_count + EPS)
    f1 = 2*p*r / (p + r + EPS)
    return p, r, f1


class TestCaseBase(TestCase):

    def setUp(self):
        self.pipeline = None
        self.test_set_data = None

    def _predict(self, limit, batch_size=512):
        test_data = self.test_set_data[:limit]
        citation_count = len(test_data)
        num_batches = int(math.ceil(citation_count / batch_size))

        predictions = []
        for idx in range(num_batches):
            batch_start = idx * batch_size
            batch_end = (idx + 1) * batch_size
            batch_inputs = test_data[batch_start:batch_end]
            batch_predictions = self.pipeline.predict(batch_inputs)
            predictions.extend(batch_predictions)
        return predictions