import json
import os.path
import pandas as pd
import pytrec_eval


WORKING_DIR="/net/intdev/pubmed_mti/ncbi/working_dir/mtix/scripts/create_test_set_descriptor_ground_truth"


def create_lookup(path):
    data = pd.read_csv(path, sep="\t", header=None)
    lookup = dict(zip(data.iloc[:,0], data.iloc[:,1]))
    return lookup


def main():
    desc_names_path =    os.path.join(WORKING_DIR, "main_heading_names.tsv")
    desc_uis_path =      os.path.join(WORKING_DIR, "main_headings.tsv")
    qrels_path =         os.path.join(WORKING_DIR, "test_set_2017-2023_qrels.tsv")
    ground_truth_path =  os.path.join(WORKING_DIR, "test_set_2017-2023_Descriptor_Ground_Truth.json")
    test_set_data_path = os.path.join(WORKING_DIR, "test_set_data.json")

    desc_names = create_lookup(desc_names_path)
    desc_uis =   create_lookup(desc_uis_path)
    qrels = pytrec_eval.parse_qrel(open(qrels_path))
    test_set_data = json.load(open(test_set_data_path))
    data_lookup = { citation_data["uid"]: citation_data["data"] for citation_data in test_set_data}

    mti_json = []
    for q_id in qrels:
        pmid = int(q_id)
        citation_predictions = { "PMID": pmid, "text-gz-64": data_lookup[pmid],  "Indexing": [] }
        mti_json.append(citation_predictions)
        for p_id in qrels[q_id]: 
            label_id = int(p_id)
            name = desc_names[label_id]
            ui =   desc_uis[label_id]
            score = 1.
            citation_predictions["Indexing"].append({
                "Term": name, 
                "Type": "Descriptor", 
                "ID": ui, 
                "IM": "NO", 
                "Reason": f"score: {score:.3f}"})

    json.dump(mti_json, open(ground_truth_path, "wt"), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()