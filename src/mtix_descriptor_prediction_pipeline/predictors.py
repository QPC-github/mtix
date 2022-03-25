class CnnModelTopNPredictor:
    def __init__(self, tensorflow_predictor):
        self.tensorflow_predictor = tensorflow_predictor

    def predict(self, citation_data_lookup):
        citation_data_list = list(citation_data_lookup.values())
        data = { "instances": [{ key: value for key, value in citation_data.items() if key not in ["pmid", "journal_title"] } for citation_data in citation_data_list] }
        response = self.tensorflow_predictor.predict(data)
        predictions = response["predictions"]
        pmids = [citation_data["pmid"] for citation_data in citation_data_list]
        top_results = { str(pmid): { str(int(desc_id)): float(score) for desc_id, score in citation_top_results} for pmid, citation_top_results in zip(pmids, predictions)}
        return top_results

class PointwiseModelTopNPredictor:
    def __init__(self, desc_name_lookup, top_n):
        self.desc_name_lookup = desc_name_lookup
        self.top_n = top_n

    def predict(self, citation_data, top_results):
        top_results = None
        return top_results

class ListwiseModelTopNPredictor:
    def __init__(self, desc_name_lookup, top_n):
        self.desc_name_lookup = desc_name_lookup
        self.top_n = top_n

    def predict(self, citation_data, top_results):
        top_results = None
        return top_results