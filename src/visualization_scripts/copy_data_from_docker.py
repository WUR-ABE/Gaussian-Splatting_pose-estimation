#%% Code to retrieve predictions in docker

import mmcv
import os
import pickle

raw_pkls_folder = "data_disk/testing/raw_pkls"
for predictions_pkl in os.listdir(raw_pkls_folder):

    predictions = mmcv.load(os.path.join(raw_pkls_folder, predictions_pkl))
    prediction_arrays = []

    for pred in predictions:
        prediction_arrays += [pred["boxes_3d"].tensor.numpy()]

    # Save to pickle
    save_folder = f"data_disk/testing/testing_dataset/{predictions_pkl.split('.')[0]}/predictions.pkl"
    pickle.dump(prediction_arrays, open(save_folder, "wb"))