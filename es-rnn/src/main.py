from utils import *

h1_train, h1_test = process_and_split_data("../data/Hourly-train.csv", "../data/Hourly-test.csv", 'H1')

print(h1_train)
