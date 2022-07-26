import json
import os.path
import pandas as pd


WORKING_DIR="/home/raear/working_dir/mtix/scripts/create_test_set_subheading_ground_truth"


def create_lookup(path):
    data = pd.read_csv(path, sep="\t", header=None)
    lookup = dict(zip(data.iloc[:,0], data.iloc[:,1]))
    return lookup


def main():
    desc_names_path =       os.path.join(WORKING_DIR, "main_heading_dui_name_mapping.tsv")
    ground_truth_path =     os.path.join(WORKING_DIR, "test_set_2017-2022_Subheading_Ground_Truth.json")
    subheading_names_path = os.path.join(WORKING_DIR, "subheading_names.tsv")
    test_set_data_path =    os.path.join(WORKING_DIR, "test_set_data.json")
    test_set_path =         os.path.join(WORKING_DIR, "test_set.jsonl")

    desc_names =       create_lookup(desc_names_path)
    subheading_names = create_lookup(subheading_names_path)
    test_set_data = json.load(open(test_set_data_path))
    test_set = [json.loads(line) for line in open(test_set_path)]

    data_lookup = { citation_data["uid"]: citation_data["data"] for citation_data in test_set_data}

    mti_json = []
    for example in test_set:
        pmid = example["pmid"]
        citation_predictions = { "PMID": pmid, "text-gz-64": data_lookup[pmid], "Indexing": [] }
        mti_json.append(citation_predictions)
        for dui, qui_list in example["mesh_headings"]:
            descriptor_prediction = { "Term": desc_names[dui], "Type": "Descriptor", "ID": dui, "IM": "NO", "Reason": f"score: {1.:.3f}", "Subheadings": []}
            citation_predictions["Indexing"].append(descriptor_prediction)
            for qui in qui_list:
                descriptor_prediction["Subheadings"].append({
                        "ID": qui,
                        "IM": "NO",
                        "Name": subheading_names[qui],
                        "Reason": f"score: {1.:.3f}"})

    json.dump(mti_json, open(ground_truth_path, "wt"), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()