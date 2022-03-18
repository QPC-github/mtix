from .utils import apply_threshold, avg_top_results, create_query_lookup 


class DescriptorPredictionPipeline():
    def __init__(self, config, cnn_model_top_n_predictor, pointwise_model_top_n_predictor, listwise_model_top_n_predictor, results_formatter):
        self.config = config
        self.input_data_parser = InputDataParser()
        self.cnn_model_top_n_predictor = cnn_model_top_n_predictor
        self.pointwise_model_top_n_predictor = pointwise_model_top_n_predictor
        self.listwise_model_top_n_predictor = listwise_model_top_n_predictor
        self.results_formatter = results_formatter

    def predict(self, input_data):
        citation_data = self.input_data_parser.parse(input_data)
        query_lookup = create_query_lookup(citation_data)
        cnn_results = self.cnn_model_top_n_predictor.predict(citation_data)
        pointwise_results = self.pointwise_model_top_n_predictor.predict(query_lookup, cnn_results)
        pointwsie_avg_results = avg_top_results(cnn_results, pointwise_results)
        listwise_results = self.listwise_model_top_n_predictor.predict(query_lookup, pointwsie_avg_results)
        listwise_avg_results = avg_top_results(pointwsie_avg_results, listwise_results)
        results = apply_threshold(listwise_avg_results, self.config["threshold"])
        predictions = self.results_formatter.format(results)
        return predictions


class InputDataParser():
    def __init__(self):
        pass
    
    def parse(self, input_data):
        citation_data = [
            { "pmid": 1,
              "title": "",
              "abstract": "",
              "pub_year": "",
              "journal name": "",
              "journal_nlmid": "",}
        ]
        return citation_data


class MtiJsonResultsFormatter():
    def __init__(self, dui_lookup):
        self.dui_lookup = dui_lookup

    def format(results):
        mti_json_object = None
        return mti_json_object