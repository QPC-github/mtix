POINTWISE_QUERY_TEMPLATE = "2017-2021|{journal_title}|{title}|{abstract}"


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
    def __init__(self, huggingface_predictor, desc_name_lookup, top_n):
        self.huggingface_predictor = huggingface_predictor
        self.desc_name_lookup = desc_name_lookup
        self.top_n = top_n

    def predict(self, citation_data_lookup, top_results):
        inputs, pmid_list, label_id_list = self.create_inputs(citation_data_lookup, top_results)
        data = { "inputs": inputs, "parameters": {"max_length": 512, "padding": "max_length", "truncation": "longest_first", "return_all_scores": True, }, }
        pointwise_result_list = self.huggingface_predictor.predict(data)
        pointwise_result_list = [label_score for label_score_list in pointwise_result_list for label_score in label_score_list if label_score["label"] == "LABEL_1"]

        pointwise_top_results = {}
        pmid_count = len(pmid_list)
        for idx in range(pmid_count):
            pmid = pmid_list[idx]
            label_id = label_id_list[idx]
            score = float(pointwise_result_list[idx]["score"])
            q_id = str(pmid)
            p_id = str(label_id)
            if q_id not in pointwise_top_results:
                pointwise_top_results[q_id] = {}
            pointwise_top_results[q_id][p_id] = score
                
        return pointwise_top_results

    def create_inputs(self, citation_data_lookup, top_results):
        inputs = []
        pmid_list = []
        label_id_list = []
        for q_id in top_results:
            pmid = int(q_id)
            data = citation_data_lookup[pmid]
            for p_id, _ in sorted(top_results[q_id].items(), key=lambda x: x[1], reverse=True)[:self.top_n]:
                label_id = int(p_id)
                query = POINTWISE_QUERY_TEMPLATE.format(journal_title=data["journal_title"], title=data["title"], abstract=data["abstract"])
                passage = self.desc_name_lookup[label_id]
                inputs.append([[query, passage]])
                pmid_list.append(pmid)
                label_id_list.append(label_id)
        return inputs, pmid_list, label_id_list


class ListwiseModelTopNPredictor:
    def __init__(self, desc_name_lookup, top_n):
        self.desc_name_lookup = desc_name_lookup
        self.top_n = top_n

    def predict(self, citation_data, top_results):
        top_results = None
        return top_results