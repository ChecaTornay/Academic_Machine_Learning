import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import Helpers as hp

#############----------Dataset dowload-----------#############
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

#############-------Data Loading-------#############
def load_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

###########--------Test Set Creation------##########
#random way split
def split_train_test(data, test_ratio):
    shufled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indicesd = shufled_indices[:test_set_size]
    train_indices = shufled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indicesd]
#--------------------------------------------------------------------------------
#hash identifier split
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_rario, id_colimn):
    ids = data[id_colimn]
    in_test_set = ids.apply(lambda id_:test_set_check(id_, test_rario))
    return data.loc[~in_test_set], data.loc[in_test_set]
#--------------------------------------------------------------------------------
#scikit-learn random split
def sklearn_random_split(data, size, seed):
    return train_test_split(data, test_size=size, random_state=seed)
#--------------------------------------------------------------------------------
#sklear stratifie split
def sklearn_stratified_split(housing, housing_income_cat, n_split=1, test_size=0.2, random_state=42):
    split = StratifiedShuffleSplit(n_splits=n_split, test_size=test_size, random_state=random_state)
    for train_index, test_index in split.split(housing, housing_income_cat["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    return strat_train_set, strat_test_set

###########--------Data Preprocesing--------##########
def data_cut(data):
    housing["income_cat"] = pd.cut(data["median_income"], bins=[0., 1.5, 3, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
    housing["income_cat"].hist()
    return housing

if __name__ == "__main__":
    housing = load_data()

    # train_set, test_set= sklearn_random_split(housing, 0.2, 42)
    # hp.show_data(housing, True, True, True, True)
    # hp.show_hist(housing)
    housing_cut = data_cut(housing)
    strat_train_set, strat_test_set = sklearn_stratified_split(housing, housing_cut)
    print(strat_test_set.value_counts/len(strat_test_set))

    
