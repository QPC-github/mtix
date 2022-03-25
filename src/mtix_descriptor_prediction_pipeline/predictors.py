class CnnModelTopNPredictor:
    def __init__(self, top_n):
        self.top_n = top_n

    def predict(self, citation_data):
        top_results = None
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