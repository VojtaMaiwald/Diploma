
import glob
import numpy as np
import re

TRAIN_LABELS_PATH = "C:/Users/Vojta/DiplomaProjects/AffectNet/train_set/annotations/"
TRAIN_LABELS_OUT_PATH_VAL = "C:/Users/Vojta/DiplomaProjects/AffectNet/train_set/all_labels_val.npy"
TRAIN_LABELS_OUT_PATH_ARO = "C:/Users/Vojta/DiplomaProjects/AffectNet/train_set/all_labels_aro.npy"
TRAIN_LABELS_OUT_PATH_EXP = "C:/Users/Vojta/DiplomaProjects/AffectNet/train_set/all_labels_exp.npy"
TRAIN_LABELS_OUT_PATH_LND = "C:/Users/Vojta/DiplomaProjects/AffectNet/train_set/all_labels_lnd.npy"

TEST_LABELS_PATH = "C:/Users/Vojta/DiplomaProjects/AffectNet/val_set/annotations/"
TEST_LABELS_OUT_PATH_VAL = "C:/Users/Vojta/DiplomaProjects/AffectNet/val_set/all_labels_val.npy"
TEST_LABELS_OUT_PATH_ARO = "C:/Users/Vojta/DiplomaProjects/AffectNet/val_set/all_labels_aro.npy"
TEST_LABELS_OUT_PATH_EXP = "C:/Users/Vojta/DiplomaProjects/AffectNet/val_set/all_labels_exp.npy"
TEST_LABELS_OUT_PATH_LND = "C:/Users/Vojta/DiplomaProjects/AffectNet/val_set/all_labels_lnd.npy"

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

if __name__ == "__main__":
    labels = np.array([float(np.load(label)) for label in sorted(glob.glob(TEST_LABELS_PATH + "*_val.npy"), key =  natural_keys)])
    np.save(TEST_LABELS_OUT_PATH_VAL, labels),
    print("TEST_LABELS_OUT_PATH_VAL saved")
    labels = np.array([float(np.load(label)) for label in sorted(glob.glob(TEST_LABELS_PATH + "*_aro.npy"), key =  natural_keys)])
    np.save(TEST_LABELS_OUT_PATH_ARO, labels),
    print("TEST_LABELS_OUT_PATH_ARO saved")
    labels = np.array([int(np.load(label)) for label in sorted(glob.glob(TEST_LABELS_PATH + "*_exp.npy"), key =  natural_keys)])
    np.save(TEST_LABELS_OUT_PATH_EXP, labels),
    print("TEST_LABELS_OUT_PATH_EXP saved")
    labels = np.array([np.load(label) for label in sorted(glob.glob(TEST_LABELS_PATH + "*_lnd.npy"), key =  natural_keys)])
    np.save(TEST_LABELS_OUT_PATH_LND, labels),
    print("TEST_LABELS_OUT_PATH_LND saved")

    labels = np.array([float(np.load(label)) for label in sorted(glob.glob(TRAIN_LABELS_PATH + "*_val.npy"), key =  natural_keys)])
    np.save(TRAIN_LABELS_OUT_PATH_VAL, labels)
    print("TRAIN_LABELS_OUT_PATH_VAL saved")
    labels = np.array([float(np.load(label)) for label in sorted(glob.glob(TRAIN_LABELS_PATH + "*_aro.npy"), key =  natural_keys)])
    np.save(TRAIN_LABELS_OUT_PATH_ARO, labels)
    print("TRAIN_LABELS_OUT_PATH_ARO saved")
    labels = np.array([int(np.load(label)) for label in sorted(glob.glob(TRAIN_LABELS_PATH + "*_exp.npy"), key =  natural_keys)])
    np.save(TRAIN_LABELS_OUT_PATH_EXP, labels)
    print("TRAIN_LABELS_OUT_PATH_EXP saved")
    labels = np.array([np.load(label) for label in sorted(glob.glob(TRAIN_LABELS_PATH + "*_lnd.npy"), key =  natural_keys)])
    np.save(TRAIN_LABELS_OUT_PATH_LND, labels)
    print("TRAIN_LABELS_OUT_PATH_LND saved")
