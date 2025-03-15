import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time
import datetime
import yaml

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

with open("config/common.yaml", "r") as common_params:
    common = yaml.safe_load(common_params)
    # unpack params
    conditions = common["conditions"]
    class_cols = common["class_cols"]
    n_classes = common["n_classes"]
    n_class_values = common["n_class_values"]

pred_cols = [col + "_pred" for col in class_cols]
confidence_cols = [col + "_confidence" for col in class_cols]

#####

def get_metrics(df):
    """
    Get metrics for the input data frame as a dictionary, one for each condition
    """
    output_by_class = {}
    #
    for i,(class_col,pred_col) in enumerate(zip(class_cols, pred_cols)):
        # Stats
        val_predict_pos = df[pred_col].map(lambda x: 1 if x==1 else 0).to_numpy()
        val_actual_pos = df[class_col].map(lambda x: 1 if x==1 else 0).to_numpy()
        # 
        TP = float(np.sum((val_predict_pos==1) & (val_actual_pos==1)))
        FN = float(np.sum((val_predict_pos==0) & (val_actual_pos==1)))
        FP = float(np.sum((val_predict_pos==1) & (val_actual_pos==0)))
        TN = float(np.sum((val_predict_pos==0) & (val_actual_pos==0)))
        #
        TPR, TNR, FPR, FNR = None, None, None, None
        BER, prec, recall, f1_score = None, None, None, None
        #
        if (TP+FN):
            TPR = float(TP / (TP + FN))  ### Sensitivity
        if (TN+FP):
            TNR = float(TN / (TN + FP))  ### Specificity
        if (TN+FP):
            FPR = float(FP / (TN + FP))
        if (FN+TP):
            FNR = float(FN / (FN + TP))
        if (FPR) and (FNR):
            BER = float((1/2) * (FPR + FNR))
        if (TP+FP+1):
            prec = float((TP + 1) / (TP + FP + 1))
        if (TP+FN+1):
            recall = float((TP + 1) / (TP + FN + 1))
        if (prec+recall):
            f1_score = float(2 * ((prec * recall) / (prec + recall)))
        #
        output_by_class[class_col] = {"TP":TP, "FP":FP, "FN":FN, "TN":TN,
                                      "TPR":TPR, "FPR":FPR, "FNR":FNR, "TNR":TNR, 
                                      "Precision":prec, "Recall":recall,
                                      "BER":BER, "F1 Score":f1_score}
    return output_by_class

def get_uncertainty_metrics(df):
    """
    Get metrics for the input data frame as a dictionary, one for each condition
    """
    output_by_class = {}
    #
    for i,(class_col,pred_col) in enumerate(zip(class_cols, pred_cols)):
        ### Stats
        predictP = df[pred_col].map(lambda x: 1 if x==1 else 0).to_numpy()
        actualP = df[class_col].map(lambda x: 1 if x==1 else 0).to_numpy()
        predictU = df[pred_col].map(lambda x: 1 if x==2 else 0).to_numpy()
        actualU = df[class_col].map(lambda x: 1 if x==2 else 0).to_numpy()
        
        ### Uncertainty metrics
        # The fraction of actual uncertain images predicted as uncertain
        frac_actualU_predictU  = float(
            np.sum(
                (actualU==1) & (predictU==1)
            )
            *
            (1 / np.sum(actualU==1))
        )
        # The fraction of actual uncertain images predicted as certain positive
        frac_actualU_predictCP = float(
            np.sum(
                (actualU==1) & (predictU==0) & (predictP==1)
            )
            *
            (1 / np.sum(actualU==1))
        ) 
        # The fraction of actual uncertain images predicted as certain negative
        frac_actualU_predictCN = float(
            np.sum(
                (actualU==1) & (predictU==0) & (predictP==0)
            )
            *
            (1 / np.sum(actualU==1))
        ) 
        # The fraction of actual certain positive images predicted as certain uncertain
        frac_actualCP_predictU = float(
            np.sum(
                (actualU==0) & (actualP==1) & (predictU==1)
            )
            *
            (1 / np.sum(actualU==0))
        ) 
        # The fraction of actual certain negative images predicted as uncertain
        frac_actualCN_predictU = float(
            np.sum(
                (actualU==0) & (actualP==0) & (predictU==1)
            )
            *
            (1 / np.sum(actualU==0))
        ) 
        # The fraction of actual certain images predicted as certain positive
        frac_actualCP_predictCP = float(
            np.sum(
                (predictU==0) & (actualU==0) & (predictP==1) & (actualP==1)
            )
            *
            (1 / np.sum(actualU==0))
        ) 
        # The fraction of actual certain images predicted as certain negative
        frac_actualCN_predictCN = float(
            np.sum(
                (predictU==0) & (actualU==0) & (predictP==0) & (actualP==0)
            )
            *
            (1 / np.sum(actualU==0))
        ) 
        # 
        TPU = frac_actualU_predictU
        FPU = frac_actualCP_predictU + frac_actualCN_predictU
        FPU_AP = frac_actualCP_predictU
        FPU_AN = frac_actualCN_predictU
        FNU = frac_actualU_predictCP + frac_actualU_predictCN
        TNU = frac_actualCP_predictCP + frac_actualCN_predictCN
        TNU_AP = frac_actualCP_predictCP
        TNU_AN = frac_actualCN_predictCN
        #
        if (TPU+FPU+1):
            prec = float((TPU + 1) / (TPU + FPU + 1))
        if (TPU+FNU+1):
            recall = float((TPU + 1) / (TPU + FNU + 1))
        if (prec+recall):
            f1_score = float(2 * ((prec * recall) / (prec + recall)))
        #
        output_by_class[class_col] = {"TPU":TPU, "FPU":FPU, "FNU":FNU, "TNU":TNU,
                                     "FPU_AP":FPU_AP, "FPU_AN":FPU_AN, "TNU_AP":TNU_AP, "TNU_AN":TNU_AN,
                                     "Precision":prec, "Recall":recall, "F1 Score":f1_score}
    return output_by_class

def get_all_metrics(input_file_paths_dict, include_certain_vs_uncertain=False):
    """
    Get a dictionary containing metrics at the condition level for each df in input_file_paths_dict
    Output is structured as output[df_name][analysis_type][condition][metric]

    df_name: The name of the model being analyzed
    analysis_type: What exactly is being analyzed by the metrics
        positive_vs_negative: Metrics are based on certain positive (1) and negative (0) predictions without including uncertain (2)
        certain_vs_uncertain: Metrics are based on certain (0/1) and uncertain (2) predictions without regard to accuracy
    condition: The condition of interest
    metric: The metric of interest

    certain_vs_uncertain: The radiologist-labeled datasets don't have uncertainty - set to False for these datasets
    """
    print("These models will be analyzed:")
    output_results = {}
    for df_name,path in input_file_paths_dict.items():
        print("   ", df_name)
        output_results[df_name] = {}
        df = pd.read_csv(path)
        
        # Get metrics for positive/negative predictions (uncertain not included)
        output_results[df_name]["positive_vs_negative"] = get_metrics(df)

        if include_certain_vs_uncertain:
        # Get metrics for certain/uncertain predictions 
            output_results[df_name]["certain_vs_uncertain"] = get_uncertainty_metrics(df)
    return output_results

def results_to_df(results_dict):
    """Convert dictionary of results to data frame"""
    results_list = []
    for results_df_name,results_data in results_dict.items():
        for model_name,model in results_dict[results_df_name].items():
            for metric_category,metrics in results_dict[results_df_name][model_name].items():
                for condition,cond_metrics in results_dict[results_df_name][model_name][metric_category].items():
                    for metric,value in results_dict[results_df_name][model_name][metric_category][condition].items():
                        results_list.append(
                            [results_df_name, 
                             model_name,
                             metric_category,
                             condition,
                             metric,
                             value
                            ]
                        )
    results_df = pd.DataFrame(results_list, columns=["results_df_name", "model_name", "metric_category", "condition", "metric", "value"])
    
    return results_df

############################

def get_accuracy(predictions, actual):
    """Given list-like predictions and corresponding list-like actual, return the accuracy"""
    return np.mean(np.array(predictions)==np.array(actual))

def get_metrics_byClassAndValue(df, pred_col, class_col, **kwargs):
    """
    Given:
        df: The data frame of results from the model
    Return:
        avgstats_byClass: Average metrics given at the class level
            ex: Average precision for Pneumonia
        stats_byClass_byValue_i: Metrics given at the class value level
            ex: Precision for Pneumonia when actual = 1
    * Average metrics use nanmean to ignore any NaN values
    """
    statsByClass = {}
    avgstats_byClass = {}
    stats = ["Value", "TP", "FP", "FN", "TN",
             "TPR", "FPR", "FNR", "TNR", "Precision", "Recall", "BER", "F1 Score"]
    # Stats
    avg_class_stats = np.zeros((len(stats), n_class_values))
    for i in range(n_class_values):
        val_predict_pos = df[pred_col].map(lambda x: 1 if x==i else 0).to_numpy()
        val_actual_pos = df[class_col].map(lambda x: 1 if x==i else 0).to_numpy()
        # 
        TP = np.sum((val_predict_pos==1) & (val_actual_pos==1))
        FN = np.sum((val_predict_pos==0) & (val_actual_pos==1))
        FP = np.sum((val_predict_pos==1) & (val_actual_pos==0))
        TN = np.sum((val_predict_pos==0) & (val_actual_pos==0))
        #
        TPR = TP / (TP + FN)  ### Sensitivity
        TNR = TN / (TN + FP)  ### Specificity
        FPR = FP / (TN + FP)
        FNR = FN / (FN + TP)
        #
        BER = (1/2) * (FPR + FNR)
        #
        prec = (TP + 1) / (TP + FP + 1)
        recall = (TP + 1) / (TP + FN + 1)
        f1_score = 2 * ((prec * recall) / (prec + recall))
        statsByClass[i] = {"Value":i,
                           "TP":TP, "FP":FP, "FN":FN, "TN":TN,
                           "TPR":TPR, "FPR":FPR, "FNR":FNR, "TNR":TNR, 
                           "Precision":prec, "Recall":recall,
                           "BER":BER, "F1 Score":f1_score}
        avg_class_stats[:,i] = [i, TP, FP, FN, TN, TPR, FPR, FNR, TNR, prec, recall, BER, f1_score]

    # Aggregate stats across classes
    for i,stat in enumerate(stats):
        avgstats_byClass["Avg Class " + stat] = np.nanmean(avg_class_stats[i,:])
    stats_byClass_byValue_i = pd.DataFrame.from_dict(statsByClass).T
    stats_byClass_byValue_i["Class"] = class_col
        
    return avgstats_byClass, stats_byClass_byValue_i, stats

def get_metrics_byClass(df):
    """
    Given:
        df: The data frame of results from the model
    Return:
        output_stats: A dictionary containing:
            stats_byClass_byValue: Metrics given at the class value level
                ex: Precision for Pneumonia when actual = 1
            avgstats_byClass: Average metrics given at the class level
                ex: Average precision for Pneumonia
            avgStats_byValue: Average metrics given at the value level
                ex: Average precision for all actual positive instances
        stats_byClass
    * Average metrics use nanmean to ignore any NaN values
    """
    output_stats = {}

    predictions = df[pred_cols].to_numpy()
    actual = df[class_cols].to_numpy()

    # Get accuracy
    output_stats["Overall Accuracy"] = get_accuracy(predictions, actual)

    # Get accuracy by class, as well as precision and recall at the different levels
    acc_byClass = {}
    avgstats_byClass = {}
    stats_byClass = {}
    for i,(class_col,pred_col) in enumerate(zip(class_cols, pred_cols)):
        col_predictions = df[pred_col].to_numpy()
        col_actual = df[class_col].to_numpy()
        acc_byClass[class_col] = get_accuracy(col_predictions, col_actual)
        #
        avgstats_byClass_i, stats_byClass_byValue_i, stats = get_metrics_byClassAndValue(df, 
                                                                                         pred_col, 
                                                                                         class_col
                                                                                        )
        stats_byClass[class_col] = stats_byClass_byValue_i
        avgstats_byClass[class_col] = avgstats_byClass_i

        if i==0:
            stats_byClass_byValue = stats_byClass_byValue_i.copy()
        else:
            stats_byClass_byValue = pd.concat([stats_byClass_byValue, stats_byClass_byValue_i], axis=0)

    avgstats_byValue = np.zeros((n_class_values, len(stats), n_classes))
    # Aggregate stats_byClass to get avgstats_byValue
    for i,(key) in enumerate(stats_byClass.keys() ):
        avgstats_byValue[:,:,i] = stats_byClass[key].iloc[:,:-1]

    output_stats["Stats by Class by Value"] = stats_byClass_byValue.reset_index(drop=True)
    output_stats["Avg Stats by Class"] = pd.DataFrame(avgstats_byClass)
    output_stats["Avg Stats by Value"] = pd.DataFrame(avgstats_byValue.mean(axis=2), columns=["Avg Value " + stat for stat in stats]).T

    return output_stats, stats_byClass

def get_metrics2(dfs, output_cols, **kwargs):
    """
    Get average metrics for input data frames
    Given:
        dfs: A list of data frames where the columns are the metrics, and the rows are test set instances
        output_cols: A list of output column names representing the corresponding df names
    Return:
        avgModelStats_byClass: a data frame where each row contains stats for each model, aggregated by class
        avgModelStats_byValue: a data frame where each row contains stats for each model, aggregated by value
        stats_byClass_byValue: a dictionary of dictionaries, one per model containing the unaggregated metrics
            Overall Accuracy: The averaged accuracy across all class values for the model
            Stats by Class by Value: A data frame containing the stats for each class value
            Avg Stats by Class: A data frame containing the stats aggregated by class
            Avg Stats by Value: A data frame containing the stats aggregated by value
    * Average metrics use nanmean to ignore any NaN values
    """
    if len(dfs) != len(output_cols):
        print("len(dfs) must match len(output_cols)")
        return None
    else:
        avgModelStats_byClass, avgModelStats_byValue = {}, {}
        stats_byClass_byValue = {}
        for df,output_col in zip(dfs, output_cols):
            stats_dict, stats_dict_byClass = get_metrics_byClass(df)
            df1 = stats_dict["Avg Stats by Class"].loc[["Avg Class Precision", 
                                                        "Avg Class TPR",
                                                        "Avg Class Recall", 
                                                        "Avg Class TNR",
                                                        "Avg Class F1 Score",
                                                        "Avg Class BER"],:].mean(axis=1)
            df2 = stats_dict["Avg Stats by Value"].loc[["Avg Value Precision", 
                                                        "Avg Value TPR",
                                                        "Avg Value Recall", 
                                                        "Avg Value TNR",
                                                        "Avg Value F1 Score",
                                                        "Avg Value BER"],:].mean(axis=1)
            avgModelStats_byClass[output_col] = df1
            avgModelStats_byValue[output_col] = df2
            stats_byClass_byValue[output_col] = stats_dict
            
        avgModelStats_byClass = pd.DataFrame.from_dict(avgModelStats_byClass).T
        avgModelStats_byValue = pd.DataFrame.from_dict(avgModelStats_byValue).T
        return (avgModelStats_byClass, avgModelStats_byValue, stats_byClass_byValue)

def plot_multiple_avg_metrics(df, metrics_cols, n_plots_x=None, n_plots_y=None, grid=None, figsize=(5,5)):
    if n_plots_x is None: n_plots_x = len(df.columns)
    if n_plots_y is None: n_plots_y = 1
    fig, axes = plt.subplots(n_plots_x, n_plots_y, figsize=figsize)
    for i,(ax,metric_col) in enumerate(zip(axes.flatten(), metrics_cols)):
        plot_df = df.sort_values(by=metric_col, ascending=False)
        sns.barplot(data=plot_df, y="Condition", x=metric_col, hue="Image Type", ax=ax, orient="h")
        ax.set(title=metric_col, xlim=(0,1), ylabel="", xlabel="")
        if grid:
            ax.grid(axis="x")
    plt.tight_layout()
    return fig,axes
