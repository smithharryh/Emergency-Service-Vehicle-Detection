emergency_train_dataset_path = '../Datasets/Data//Emergency/'
non_emergency_train_dataset_path = '../Datasets/Data/nonEmergency/'

import csv
import os

with open("metadata.csv", 'w')as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["filename","class_name"])
    for file_name in os.listdir(emergency_train_dataset_path):
        csvwriter.writerow([file_name, "emergency"])
    for file_name in os.listdir(non_emergency_train_dataset_path):
        csvwriter.writerow([file_name, "non_emergency"])