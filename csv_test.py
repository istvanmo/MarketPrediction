import csv
import matplotlib.pyplot as plt

close_prices = []
volumes = []
dates = []
with open('sp500.csv', newline='') as csvfile:
    sp500 = csv.reader(csvfile)
    i = False
    for row in sp500:
        if i:
            dates.append(row[0])
            close_prices.append(float(row[4]))
            volumes.append(float(row[6]))
        else:
            print(row)
            i = True
