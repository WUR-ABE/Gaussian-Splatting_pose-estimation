import json
import numpy as np
import pandas as pd
import os
import yaml
import sys

def main():
    params = yaml.safe_load(open("params.yaml"))["compute_metrics"]
    confidence_threshold = params["confidence_threshold"]

    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython compute_metrics.py validation-results-file \n")
        sys.exit(1)

    eval_path = sys.argv[1]

    # # Testing folders
    # eval_path = "results/evaluation/real_randwijk_freshnet_model/val_real_randwijk_freshnet_model_latest.txt"

    # Load validation results
    with open(eval_path, "r") as f:
        s = f.read()
        lines = s.split("\n")

    results = {}
    for line in lines[:-1]:
        vis_dict = json.loads(line.replace("\'", "\""))
        results[vis_dict["visibility_lower_bound"]] = vis_dict


    for key in results.keys():
        for iou_threshold in ["25", "50"]:
            f1_scores = []
            for i in range(len(results[key][f"apple_confidences_{iou_threshold}"])):
                f1_scores += [2 * (results[key][f"apple_precisions_{iou_threshold}"][i] * results[key][f"apple_recalls_{iou_threshold}"][i]) / (results[key][f"apple_precisions_{iou_threshold}"][i] + results[key][f"apple_recalls_{iou_threshold}"][i] + np.finfo(float).eps)]
            results[key][f"apple_f1_{iou_threshold}"] = f1_scores

    results_for_plotting = [[
        "visibility", 
        "recall_25", "precision_25", "f1_25", 
        "recall_50", "precision_50", "f1_50", 
    ]]
    results_for_metrics = {}
    for key in results.keys():
        result_for_plotting = [key]
        result_for_metrics = {}
        for iou_threshold in ["25", "50"]:
            high_confidence_indices = np.where(np.array(results[key][f"apple_confidences_{iou_threshold}"]) >= confidence_threshold)[0]
            i = high_confidence_indices[-1]

            result_for_plotting.extend([
                results[key][f"apple_recalls_{iou_threshold}"][i],
                results[key][f"apple_precisions_{iou_threshold}"][i],
                results[key][f"apple_f1_{iou_threshold}"][i],
            ])

            result_for_metrics[f"recall_{iou_threshold}"] = results[key][f"apple_recalls_{iou_threshold}"][i]
            result_for_metrics[f"precision_{iou_threshold}"] = results[key][f"apple_precisions_{iou_threshold}"][i]
            result_for_metrics[f"f1_{iou_threshold}"] = results[key][f"apple_f1_{iou_threshold}"][i]

        results_for_plotting.append(result_for_plotting)
        results_for_metrics[f"visibility_{key}"] = result_for_metrics

    
    # Determine output path
    output_path = "dvclive/val_" + eval_path.split("/")[-2]

    # Make dir if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Write results for plotting to DF and file
    df = pd.DataFrame(results_for_plotting[1:], columns=results_for_plotting[0])
    df.to_csv(output_path + "/results.csv", index=False)

    # Write results for metrics to json
    with open(output_path + "/results.json", "w") as f:
        json.dump(results_for_metrics, f, indent=4)



if __name__ == "__main__":
    main()