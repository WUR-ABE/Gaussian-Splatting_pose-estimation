#%% Imports
import open3d as o3d
import numpy as np
import scipy.io as sio
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from multiprocessing.pool import ThreadPool
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

os.chdir("/home/gs-lfd/dvc-5d-apple-pose-estimation")

# Set the fonttype to TrueType
mpl.rcParams["text.usetex"] = True

# Font sizes
SMALL_SIZE = 11
MEDIUM_SIZE = 13
BIGGER_SIZE = 15

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# Set background as white
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
# Set grid style
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.linestyle'] = '--'
# Set axis color
plt.rcParams['axes.edgecolor'] = 'black'



#%% Load results of visibility experiment

experiment_path = "data_disk/testing/testing_visibility/"

gs_results = {}
real_results = {}
for experiment_name in os.listdir(experiment_path):
    for file_name in os.listdir(os.path.join(experiment_path, experiment_name)):
        if file_name.endswith(".txt"):
            with open(os.path.join(experiment_path, experiment_name, file_name), "r") as f:
                s = f.read()
                lines = s.split("\n")

            results = {}
            for line in lines[:-1]:
                vis_dict = json.loads(line.replace("\'null\'", "null").replace("\'", "\""))
                # Convert each entry except "visibility_lower_bound" and all AP's to arrays
                for key in vis_dict.keys():
                    if key != "visibility_lower_bound" and key != "gt_counts" and "AP_" not in key:
                        vis_dict[key] = np.asarray(vis_dict[key], dtype=float)
                results[vis_dict["visibility_lower_bound"]] = vis_dict

            if file_name.startswith("val_randwijk"):
                gs_results[experiment_name.split("_")[1]] = results
            elif file_name.startswith("val_real_randwijk"):
                real_results[experiment_name.split("_")[1]] = results

#%% Use bootstrapping to calculate confidence intervals for precision, recall and F1

def bootstrapped_performance_calculator(outputs):
    performance_indicators = {}

    for key in outputs.keys():
        performance_indicators[key] = {}
        sum_gts = sum(outputs[key]['gt_counts'].values())
        performance_indicators[key]['gts_sum'] = sum_gts
        for iou_threshold in ["_25", "_50"]:
            confidence_order = np.argsort(-outputs[key]["apple_confidences"+iou_threshold])

            # Sort all model predictions
            confidences = outputs[key]["apple_confidences"+iou_threshold][confidence_order]
            true_positives = outputs[key]["apple_t_positives"+iou_threshold][confidence_order]
            false_positives = outputs[key]["apple_f_positives"+iou_threshold][confidence_order]
            neutralized_false_positives = outputs[0.0]["apple_f_positives"+iou_threshold][confidence_order]
            neutralized_true_positives = outputs[0.0]["apple_t_positives"+iou_threshold][confidence_order]
            vect_angle_errors = outputs[key]["apple_VA"+iou_threshold][confidence_order]
            quaternion_dists = outputs[key]["apple_QD"+iou_threshold][confidence_order]
            roll_errors = outputs[key]["apple_RE"+iou_threshold][confidence_order]
            pitch_errors = outputs[key]["apple_PE"+iou_threshold][confidence_order]
            yaw_errors = outputs[key]["apple_YE"+iou_threshold][confidence_order]
            euclidean_dists = outputs[key]["apple_ED"+iou_threshold][confidence_order]

            # Calculate precision and recall from highest to lowest confidence
            tp_sum = np.cumsum(true_positives)
            fp_sum = np.cumsum(false_positives)
            recalls = tp_sum / sum_gts
            precisions = tp_sum / np.maximum(tp_sum + fp_sum, np.finfo(np.float64).eps)

            # Calculate f1_scores
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + np.finfo(np.float64).eps)

            max_f1_confidence = confidences[np.argmax(f1_scores)]
            f1_score = np.max(f1_scores)

            # Calculate precision and recall at each confidence level with neutral positives
            ntp_sum = np.cumsum(neutralized_true_positives)
            nfp_sum = np.cumsum(neutralized_false_positives)
            neutral_precisions = tp_sum / np.maximum(tp_sum + nfp_sum, np.finfo(np.float64).eps)
            double_neutral_precisions = ntp_sum / np.maximum(ntp_sum + nfp_sum, np.finfo(np.float64).eps)

            # Calculate neutral f1 scores
            neutral_f1_scores = 2 * (neutral_precisions * recalls) / (neutral_precisions + recalls + np.finfo(np.float64).eps)
            double_neutral_f1_scores = 2 * (double_neutral_precisions * recalls) / (double_neutral_precisions + recalls + np.finfo(np.float64).eps)

            neutral_max_f1_confidence = confidences[np.argmax(neutral_f1_scores)]
            double_neutral_max_f1_confidence = confidences[np.argmax(double_neutral_f1_scores)]
            neutral_f1_score = np.max(neutral_f1_scores)
            double_neutral_f1_score = np.max(double_neutral_f1_scores)
            
            desired_predictions = confidences > max_f1_confidence

            mean_vect_angle_error = np.nanmean(vect_angle_errors[desired_predictions])
            mean_quaternion_dist = np.nanmean(quaternion_dists[desired_predictions])
            mean_roll_error = np.nanmean(roll_errors[desired_predictions])
            mean_pitch_error = np.nanmean(pitch_errors[desired_predictions])
            mean_yaw_error = np.nanmean(yaw_errors[desired_predictions])
            mean_euclidean_dist = np.nanmean(euclidean_dists[desired_predictions])

            # Store the performance indicators
            performance_indicators[key].update({
                "max_f1_confidence"+iou_threshold: max_f1_confidence,
                "f1_score"+iou_threshold: f1_score,
                "neutral_max_f1_confidence"+iou_threshold: neutral_max_f1_confidence,
                "neutral_f1_score"+iou_threshold: neutral_f1_score,
                "double_neutral_max_f1_confidence"+iou_threshold: double_neutral_max_f1_confidence,
                "double_neutral_f1_score"+iou_threshold: double_neutral_f1_score,
                "mean_vect_angle_error"+iou_threshold: mean_vect_angle_error,
                "mean_quaternion_dist"+iou_threshold: mean_quaternion_dist,
                "mean_roll_error"+iou_threshold: mean_roll_error,
                "mean_pitch_error"+iou_threshold: mean_pitch_error,
                "mean_yaw_error"+iou_threshold: mean_yaw_error,
                "mean_euclidean_dist"+iou_threshold: mean_euclidean_dist,
            })

    return performance_indicators

def build_subset(results, sample_ids, iter_columns):
        outputs = {}
        sample_ids = np.array(sample_ids)
        sample_ids_str = set(map(str, sample_ids))
        for key in results.keys():
            outputs[key] = {}
            # Precompute image ids arrays for both thresholds
            image_ids_25 = np.asarray(results[key]["apple_image_ids_25"])
            image_ids_50 = np.asarray(results[key]["apple_image_ids_50"])
            for iou_threshold, image_ids in [("_25", image_ids_25), ("_50", image_ids_50)]:
                column = 'apple_image_ids'
                # Assign output_sample_id for each sample_id occurrence
                temp_outputs_list = np.empty(0, dtype=int)
                target_ids = np.empty(0, dtype=int)
                for output_sample_id, sample_id in enumerate(sample_ids):
                    mask = image_ids == sample_id
                    if np.any(mask):
                        temp_outputs_list = np.concatenate([temp_outputs_list, np.full(np.sum(mask), output_sample_id, dtype=int)])
                        target_ids = np.concatenate([target_ids, np.argwhere(mask).flatten()])
                outputs[key][column + iou_threshold] = temp_outputs_list
                for column in iter_columns:
                    arr = np.asarray(results[key][column + iou_threshold])
                    # Use boolean mask for fast selection
                    outputs[key][column + iou_threshold] = arr[target_ids]
            # Update the GT dict
            gt_dict = {output_sample_id: results[key]["gt_counts"][str(sample_id)] for output_sample_id, sample_id in enumerate(sample_ids) if str(sample_id) in results[key]["gt_counts"]}
            outputs[key]["gt_counts"] = gt_dict
        return outputs

def build_and_process_subset(results, sample_ids, iter_columns):
    """
    Build a subset of the results based on the sample ids and process it.
    """
    outputs = build_subset(results, sample_ids, iter_columns)
    return bootstrapped_performance_calculator(outputs)


iter_columns = [
    'apple_confidences', 
    'apple_t_positives', 
    'apple_f_positives', 
    'apple_VA', 
    'apple_QD', 
    'apple_RE', 
    'apple_PE', 
    'apple_YE', 
    'apple_ED',
]

# Testing sample ids
sample_ids = [2, 0, 2]


# Set seed for np random
np.random.seed(42)

bootstrapped_results = {}
result_types = ["gs", "real"]
for iter_exp_results, exp_results in enumerate([gs_results, real_results]):
    bootstrapped_results[result_types[iter_exp_results]] = {}
    for visibility_lower_bound_setting in exp_results.keys():
        print(f"Processing {result_types[iter_exp_results]} results for visibility lower bound: {visibility_lower_bound_setting}")
        curr_results = exp_results[visibility_lower_bound_setting]

        unsampled_ids = list(curr_results[list(curr_results.keys())[0]]["gt_counts"].keys())
        # Convert to integers
        unsampled_ids = [int(unsampled_id) for unsampled_id in unsampled_ids]

        # Perform sampling
        bootstrap_sample_id_sets = [
            np.random.choice(
                unsampled_ids, 
                size=len(unsampled_ids), 
                replace=True,
            ) for _ in range(100)
        ]

        n_cores = int(os.cpu_count())
        with ThreadPool(n_cores) as pool:
            output_results = list(pool.starmap(
                build_and_process_subset, 
                [
                    (
                        curr_results, sample_ids, iter_columns
                    ) for sample_ids in bootstrap_sample_id_sets
                ]
            ))

        # Convert the list of dictionaries to a single df with columns for each metric and visibility
        performance_indicators = {}
        for i, output in enumerate(output_results):
            for key in output.keys():
                for metric in output[key].keys():
                    combined_key = f"{key}_{metric}"
                    if combined_key not in performance_indicators:
                        performance_indicators[combined_key] = []
                    performance_indicators[combined_key].append(output[key][metric])

        df_performance_indicators = pd.DataFrame(performance_indicators)
        # Place in dict
        bootstrapped_results[result_types[iter_exp_results]][visibility_lower_bound_setting] = df_performance_indicators

# %% Save the bootstrapped results to a file

test_visibilities = [
    0.0, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5
]

# Convert dictionary to a DataFrame
results_df = pd.DataFrame()
for result_type in bootstrapped_results.keys():
    for visibility_lower_bound_setting in bootstrapped_results[result_type].keys():
        df_curr_results = bootstrapped_results[result_type][visibility_lower_bound_setting]
        for test_visibility in test_visibilities:
            df_curr_results_filtered = copy.deepcopy(df_curr_results.filter(like=f"{test_visibility}_"))
            df_curr_results_filtered.columns = ["_".join(col.split("_")[1:]) for col in df_curr_results_filtered.columns]
            df_curr_results_filtered["test_visibility"] = test_visibility
            df_curr_results_filtered["visibility_lower_bound"] = float(visibility_lower_bound_setting)/100.
            df_curr_results_filtered["result_type"] = result_type
            results_df = pd.concat([results_df, df_curr_results_filtered], ignore_index=True)

# Save the results to a CSV file
output_path = "data_disk/testing/testing_visibility/bootstrapped_results.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
results_df.to_csv(output_path, index=False)

#%% Import the results from the CSV file
results_df = pd.read_csv("data_disk/testing/testing_visibility/bootstrapped_results.csv")
results_df["test_occlusion"] = 1.0 - results_df["test_visibility"]
results_df["occlusion_upper_bound"] = 1.0 - results_df["visibility_lower_bound"]

#%% Determine the significant differences between models trained at different visibility lower bounds

# Perform ANOVA to determine significant differences in double_neutral_f1_score_50 at each test_occlusion
from scipy.stats import f_oneway

boundary_p_value = 0.05

anova_results = []
group_results = []

for result_type in results_df["result_type"].unique():
    for test_occlusion in results_df["test_occlusion"].unique():
        df_subset = results_df[
            (results_df["result_type"] == result_type) &
            (results_df["test_occlusion"] == test_occlusion)
        ]
        # Sort the occlusion_upper_bound values by descending order of mean double_neutral_f1_score_50
        sorted_occlusion_bounds = np.array(df_subset.groupby("occlusion_upper_bound")["double_neutral_f1_score_50"].mean().sort_values(ascending=False).index)
        # Create a list of groups for ANOVA
        # First repeat the best performing group to compare all other groups against it
        groups_a = [df_subset[df_subset["occlusion_upper_bound"] == sorted_occlusion_bounds[0]]["double_neutral_f1_score_50"].values for ob in sorted_occlusion_bounds]
        groups_a = np.asarray(groups_a)
        # Then the actual groups
        groups_b = [df_subset[df_subset["occlusion_upper_bound"] == ob]["double_neutral_f1_score_50"].values for ob in sorted_occlusion_bounds]
        groups_b = np.asarray(groups_b)

        stat, pvalue = f_oneway(groups_a.T, groups_b.T)
        ordered_groups = {
            "result_type": result_type,
            "test_occlusion": test_occlusion,
            1: sorted_occlusion_bounds[pvalue > boundary_p_value],
        }
        anova_results.append({
            "result_type": result_type,
            "test_occlusion": test_occlusion,
            "best_train_occlusion": sorted_occlusion_bounds[pvalue > boundary_p_value],
            "groups": sorted_occlusion_bounds,
            "F_statistic": stat,
            "p_value": pvalue
        })

        # Determine the additional groups
        group_id = 2
        remaining_groups = sorted_occlusion_bounds[pvalue <= boundary_p_value]
        while len(remaining_groups) > 0:
            groups_a = [df_subset[df_subset["occlusion_upper_bound"] == remaining_groups[0]]["double_neutral_f1_score_50"].values for ob in remaining_groups]
            groups_a = np.asarray(groups_a)
            groups_b = [df_subset[df_subset["occlusion_upper_bound"] == ob]["double_neutral_f1_score_50"].values for ob in remaining_groups]
            groups_b = np.asarray(groups_b)

            stat, pvalue = f_oneway(groups_a.T, groups_b.T)
            ordered_groups[group_id] = remaining_groups[pvalue > boundary_p_value]
            remaining_groups = remaining_groups[pvalue <= boundary_p_value]
            group_id += 1

        group_results.append(ordered_groups)

anova_df = pd.DataFrame(anova_results)
groups_df = pd.DataFrame(group_results)

# Save the anova results to a CSV file
output_anova_path = "data_disk/testing/testing_visibility/anova_best_occlusion_upper_bounds.csv"
os.makedirs(os.path.dirname(output_anova_path), exist_ok=True)
anova_df.to_csv(output_anova_path, index=False)

#%% Convert the groups to a rank per train occlusion
rank_df = pd.DataFrame(
    columns=["result_type", "test_occlusion", 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
)

for i, row in groups_df.iterrows():
    rank_row = {
        "result_type": row["result_type"],
        "test_occlusion": row["test_occlusion"],
    }
    for rank, group in enumerate(row.drop(["result_type", "test_occlusion"])):
        if isinstance(group, np.ndarray):
            for occlusion in group:
                rank_row[occlusion] = rank + 1
    rank_df = pd.concat([rank_df, pd.DataFrame(rank_row, index=[0])], ignore_index=True)

# Group by result type and calculate mean of ranks
mean_rank_df = rank_df.groupby("result_type").mean().reset_index()

#%% Anove for all test occlusions together
train_anova_results = []
train_group_results = []

for result_type in results_df["result_type"].unique():
    df_subset = results_df[
        (results_df["result_type"] == result_type)
    ]
    # Sort the occlusion_upper_bound values by descending order of mean double_neutral_f1_score_50
    sorted_occlusion_bounds = np.array(df_subset.groupby("occlusion_upper_bound")["double_neutral_f1_score_50"].mean().sort_values(ascending=False).index)
    # Create a list of groups for ANOVA
    # First repeat the best performing group to compare all other groups against it
    groups_a = [df_subset[df_subset["occlusion_upper_bound"] == sorted_occlusion_bounds[0]]["double_neutral_f1_score_50"].values for ob in sorted_occlusion_bounds]
    groups_a = np.asarray(groups_a)
    # Then the actual groups
    groups_b = [df_subset[df_subset["occlusion_upper_bound"] == ob]["double_neutral_f1_score_50"].values for ob in sorted_occlusion_bounds]
    groups_b = np.asarray(groups_b)

    stat, pvalue = f_oneway(groups_a.T, groups_b.T)
    ordered_groups = {
        "result_type": result_type,
        "test_occlusion": test_occlusion,
        1: sorted_occlusion_bounds[pvalue > boundary_p_value],
    }
    train_anova_results.append({
        "result_type": result_type,
        "test_occlusion": test_occlusion,
        "best_train_occlusion": sorted_occlusion_bounds[pvalue > boundary_p_value],
        "groups": sorted_occlusion_bounds,
        "F_statistic": stat,
        "p_value": pvalue
    })

    # Determine the additional groups
    group_id = 2
    remaining_groups = sorted_occlusion_bounds[pvalue <= boundary_p_value]
    while len(remaining_groups) > 0:
        groups_a = [df_subset[df_subset["occlusion_upper_bound"] == remaining_groups[0]]["double_neutral_f1_score_50"].values for ob in remaining_groups]
        groups_a = np.asarray(groups_a)
        groups_b = [df_subset[df_subset["occlusion_upper_bound"] == ob]["double_neutral_f1_score_50"].values for ob in remaining_groups]
        groups_b = np.asarray(groups_b)

        stat, pvalue = f_oneway(groups_a.T, groups_b.T)
        ordered_groups[group_id] = remaining_groups[pvalue > boundary_p_value]
        remaining_groups = remaining_groups[pvalue <= boundary_p_value]
        group_id += 1

    train_group_results.append(ordered_groups)

train_anova_df = pd.DataFrame(train_anova_results)
train_groups_df = pd.DataFrame(train_group_results)

#%% Group the results by visibility lower bound and result type and test visibility
def percentile(n):
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'percentile_{:02.0f}'.format(n*100)
    return percentile_

grouped_results_df = results_df.groupby(
    ["result_type", "occlusion_upper_bound", "test_occlusion"]
).agg(["mean", percentile(0.025), percentile(0.975), percentile(0.5)]).reset_index()

# Get visibility lower bound of model with best performance in double_neutral_f1_score_50 
# for each test visibility and result type
model_ranking = {}
model_performance = {}
for train_occlusion in grouped_results_df.occlusion_upper_bound.unique():
    model_ranking[train_occlusion] = pd.DataFrame()
    model_performance[train_occlusion] = pd.DataFrame()

df_best_occlusion_upper_bounds = pd.DataFrame()

for result_type in grouped_results_df.result_type.unique():
    for test_occlusion in grouped_results_df.test_occlusion.unique():
        df_curr_results = grouped_results_df[
            (grouped_results_df.result_type == result_type) & 
            (grouped_results_df.test_occlusion == test_occlusion)
        ]
        mean_best_row = df_curr_results.loc[df_curr_results[("double_neutral_f1_score_50", "mean")].idxmax()]
        percentile_025_best_row = df_curr_results.loc[df_curr_results[("double_neutral_f1_score_50", "percentile_02")].idxmax()]
        percentile_975_best_row = df_curr_results.loc[df_curr_results[("double_neutral_f1_score_50", "percentile_98")].idxmax()]
        percentile_50_best_row = df_curr_results.loc[df_curr_results[("double_neutral_f1_score_50", "percentile_50")].idxmax()]
        row_dict = {
            "result_type": result_type,
            "test_occlusion": test_occlusion,
            "mean": mean_best_row["occlusion_upper_bound"].values[0],
            "percentile_025": percentile_025_best_row["occlusion_upper_bound"].values[0],
            "percentile_975": percentile_975_best_row["occlusion_upper_bound"].values[0],
            "percentile_50": percentile_50_best_row["occlusion_upper_bound"].values[0],
        }
        df_best_occlusion_upper_bounds = pd.concat(
            [df_best_occlusion_upper_bounds, pd.DataFrame(row_dict, index=[0])], 
            ignore_index=True
        )

        sorted_mean_best_row = df_curr_results.sort_values(by=("double_neutral_f1_score_50", "mean"), ascending=False).occlusion_upper_bound.reset_index(drop=True)
        sorted_percentile_025_best_row = df_curr_results.sort_values(by=("double_neutral_f1_score_50", "percentile_02"), ascending=False).occlusion_upper_bound.reset_index(drop=True)
        sorted_percentile_975_best_row = df_curr_results.sort_values(by=("double_neutral_f1_score_50", "percentile_98"), ascending=False).occlusion_upper_bound.reset_index(drop=True)
        sorted_percentile_50_best_row = df_curr_results.sort_values(by=("double_neutral_f1_score_50", "percentile_50"), ascending=False).occlusion_upper_bound .reset_index(drop=True)

        for train_occlusion in df_curr_results.occlusion_upper_bound.unique():
            rank_row_dict = {
                "result_type": result_type,
                "test_occlusion": test_occlusion,
                "train_occlusion": train_occlusion,
                "mean": sorted_mean_best_row[sorted_mean_best_row == train_occlusion].index[0],
                "percentile_025": sorted_percentile_025_best_row[sorted_percentile_025_best_row == train_occlusion].index[0],
                "percentile_975": sorted_percentile_975_best_row[sorted_percentile_975_best_row == train_occlusion].index[0],
                "percentile_50": sorted_percentile_50_best_row[sorted_percentile_50_best_row == train_occlusion].index[0],
            }
            model_ranking[train_occlusion] = pd.concat(
                [model_ranking[train_occlusion], pd.DataFrame(rank_row_dict, index=[0])], 
                ignore_index=True
            )

            # Store the performance for each train visibility
            score_row_dict = {
                "result_type": result_type,
                "test_occlusion": test_occlusion,
                "train_occlusion": train_occlusion,
                "mean": df_curr_results.loc[df_curr_results.occlusion_upper_bound == train_occlusion, ("double_neutral_f1_score_50", "mean")].values[0],
                "percentile_025": df_curr_results.loc[df_curr_results.occlusion_upper_bound == train_occlusion, ("double_neutral_f1_score_50", "percentile_02")].values[0],
                "percentile_975": df_curr_results.loc[df_curr_results.occlusion_upper_bound == train_occlusion, ("double_neutral_f1_score_50", "percentile_98")].values[0],
                "percentile_50": df_curr_results.loc[df_curr_results.occlusion_upper_bound == train_occlusion, ("double_neutral_f1_score_50", "percentile_50")].values[0],
            }
            model_performance[train_occlusion] = pd.concat(
                [model_performance[train_occlusion], pd.DataFrame(score_row_dict, index=[0])], 
                ignore_index=True
            )

# Convert the model ranking dictionary to a DataFrame
df_model_ranking = pd.DataFrame()
for train_occlusion in model_ranking.keys():
    df_curr_ranking = model_ranking[train_occlusion]
    df_curr_ranking["train_occlusion"] = train_occlusion
    df_model_ranking = pd.concat([df_model_ranking, df_curr_ranking], ignore_index=True)

grouped_model_ranking = df_model_ranking.groupby(
    ["result_type", "train_occlusion"]
).agg(
    {
        "mean": ["mean"],
        "percentile_025": ["mean"],
        "percentile_975": ["mean"],
        "percentile_50": ["mean"],
    }
).reset_index()

# Convert the model performance dictionary to a DataFrame
df_model_performance = pd.DataFrame()
for train_occlusion in model_performance.keys():
    df_curr_performance = model_performance[train_occlusion]
    df_curr_performance["train_occlusion"] = train_occlusion
    df_model_performance = pd.concat([df_model_performance, df_curr_performance], ignore_index=True)

grouped_model_performance = df_model_performance.groupby(
    ["result_type", "train_occlusion"]
).agg(
    {
        "mean": ["mean"],
        "percentile_025": ["mean"],
        "percentile_975": ["mean"],
        "percentile_50": ["mean"],
    }
).reset_index()

# Save the best visibility lower bounds to a CSV file
output_best_path = "data_disk/testing/testing_visibility/best_occlusion_upper_bounds.csv"
os.makedirs(os.path.dirname(output_best_path), exist_ok=True)
df_best_occlusion_upper_bounds.to_csv(output_best_path, index=False)

# Save the model ranking to a CSV file
output_ranking_path = "data_disk/testing/testing_visibility/model_ranking.csv"
os.makedirs(os.path.dirname(output_ranking_path), exist_ok=True)
grouped_model_ranking.to_csv(output_ranking_path, index=False)

# Save the model performance to a CSV file
output_performance_path = "data_disk/testing/testing_visibility/model_performance.csv"
os.makedirs(os.path.dirname(output_performance_path), exist_ok=True)
grouped_model_performance.to_csv(output_performance_path, index=False)


# %%

column_names = [
    0.0, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5
]
# Make plots for the bootstrapped results
for result_type in results_df.result_type.unique():
    # Define the figure with two subplots
    fig1 = plt.figure(
        figsize=(3, 3),
        constrained_layout=True,
        )
    axs1 = fig1.add_subplot(1, 1, 1)
    fig2 = plt.figure(
        figsize=(3, 3),
        constrained_layout=True,
    )
    axs2 = fig2.add_subplot(1, 1, 1)

    df_curr_results = results_df[(results_df.result_type == result_type)]

    # Plot the results, with rows being repetitions, merged to show mean and confidence interval
    sns.lineplot(
        data=df_curr_results, 
        x="test_occlusion", 
        y="f1_score_50", 
        ax=axs1, 
        hue="occlusion_upper_bound",
        palette="flare",
        errorbar="pi",
        estimator="mean",
        err_style="bars",
    )

    sns.lineplot(
        data=df_curr_results, 
        x="test_occlusion", 
        y="double_neutral_f1_score_50", 
        ax=axs2, 
        hue="occlusion_upper_bound",
        palette="flare",
        errorbar="pi",
        estimator="mean",
        err_style="bars",
    )

    # Set axis ranges and labels
    axs1.set_ylim(0, 1.05)
    axs1.set_xlabel("Test max Occlusion")
    axs1.set_ylabel("F1 Score")
    axs2.set_ylim(0, 1.05)
    axs2.set_xlabel("Test max Occlusion")
    axs2.set_ylabel("Neutral F1 Score")

    # Add legend
    axs1.legend(title="Train max Occlusion")
    axs2.legend(title="Train max Occlusion")

    # Add gridlines
    axs1.grid(True, which='both', linestyle='--', linewidth=0.5)
    axs2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save the plots as PDF files
    fig1.savefig(f"data_disk/testing/testing_visibility/{result_type}_f1_score_50_vs_occlusion.pdf")
    fig2.savefig(f"data_disk/testing/testing_visibility/{result_type}_neutral_f1_score_50_vs_occlusion.pdf")

# %%
