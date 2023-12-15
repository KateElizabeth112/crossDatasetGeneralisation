import os
import pickle as pkl
import numpy as np


root_dir = "/Users/katecevora/Documents/PhD/data"
output_dir = "/Users/katecevora/Documents/PhD/results"
dataset = "TotalSegmentator"

def main():
    f = open(os.path.join(root_dir, dataset, "inference", "results_combined_cross.pkl"), 'rb')
    results_ex1 = pkl.load(f)
    f.close()

    case_id = results_ex1["case_id"].flatten()
    sex = results_ex1["sex"].flatten()
    age = results_ex1["age"].flatten()

    dice_ex1 = results_ex1["dice"].reshape((-1, np.array(results_ex1["dice"]).shape[-1]))
    hd_ex1 = np.array(results_ex1["hd"]).reshape((-1, np.array(results_ex1["hd"]).shape[-1]))
    vol_pred_ex1 = results_ex1["vol_pred"].reshape((-1, np.array(results_ex1["vol_pred"]).shape[-1]))
    vol_gt = results_ex1["vol_gt"].reshape((-1, np.array(results_ex1["vol_gt"]).shape[-1]))

    print("Done")


if __name__ == "__main__":
    main()