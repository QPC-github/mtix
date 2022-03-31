POINTWISE_QUERY_TEMPLATE = "2017-2021|{journal_title}|{title}|{abstract}"


class CnnModelTop100Predictor:
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

    def predict(self, citation_data_lookup, input_top_results):
        pmid_list = []
        label_id_list = []
        score_list = []
        for q_id in input_top_results:
            pmid = int(q_id)
            citation_data = citation_data_lookup[pmid]
            citation_top_results = input_top_results[q_id]
            citation_input_data, citation_label_id_list = self._create_input_data(citation_data, citation_top_results)
            citation_score_list = self._predict_internal(citation_input_data)
            pmid_list.extend([pmid]*self.top_n)
            label_id_list.extend(citation_label_id_list)
            score_list.extend(citation_score_list)
        output_top_results = self._create_top_results(pmid_list, label_id_list, score_list)
        return output_top_results

    def _create_input_data(self, citation_data, citation_top_results):
        inputs = []
        label_id_list = []
        for p_id, _ in sorted(citation_top_results.items(), key=lambda x: x[1], reverse=True)[:self.top_n]:
            label_id = int(p_id)
            query = POINTWISE_QUERY_TEMPLATE.format(journal_title=citation_data["journal_title"], title=citation_data["title"], abstract=citation_data["abstract"])
            passage = self.desc_name_lookup[label_id]
            inputs.append([[query, passage]])
            label_id_list.append(label_id)
        input_data = { "inputs": inputs, "parameters": {"max_length": 512, "padding": "max_length", "truncation": "longest_first", "return_all_scores": True, }, }
        return input_data, label_id_list

    def _create_top_results(self, pmid_list, label_id_list, score_list):
        top_results = {}
        result_count = len(score_list)
        for idx in range(result_count):
            pmid = pmid_list[idx]
            label_id = label_id_list[idx]
            score = score_list[idx]
            q_id = str(pmid)
            p_id = str(label_id)
            if q_id not in top_results:
                top_results[q_id] = {}
            top_results[q_id][p_id] = score
        return top_results

    def _predict_internal(self, input_data):
        score_list = self.huggingface_predictor.predict(input_data)
        score_list = [float(label_score["score"]) for label_score_list in score_list for label_score in label_score_list if label_score["label"] == "LABEL_1"]
        return score_list


class ListwiseModelTopNPredictor:
    def __init__(self, huggingface_predictor, desc_name_lookup, top_n):
        self.huggingface_predictor = huggingface_predictor
        self.desc_name_lookup = desc_name_lookup
        self.top_n = top_n

    def predict(self, citation_data_lookup, input_top_results):
        top_results = None
        return top_results