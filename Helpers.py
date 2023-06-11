import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt

def show_data(data, head = False, info = False, ocena_proximity = False, describe = False):
    if head == True:
        print(data.head())
    if info == True:
        print(data.info())
    if ocena_proximity == True:
        print(data["ocean_proximity"].value_counts())
    if describe:
        print((data.describe()))
    
def show_hist(data):
    data.hist(bins=50, figsize=(20,15))
    plt.show()