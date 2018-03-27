import csv
import matplotlib.pyplot as plt

def get_data():
    close_prices = []
    volumes = []
    dates = []
    with open('N225.csv', newline='') as csvfile:
        sp500 = csv.reader(csvfile)
        i = False
        row_d = None
        row_cp = None
        row_v = None
        for row in sp500:
            if i:
                dates.append(row[row_d])
                if row[row_cp] in ["null"]:
                    close_prices.append(close_prices[-1])
                else:
                    close_prices.append(float(row[row_cp]))

                if row[row_v] in ["null"]:
                    volumes.append(volumes[-1])
                else:
                    if float(row[row_v]) != 0:
                        volumes.append(float(row[row_v]))
                    else:
                        volumes.append(volumes[-1])
            else:
                row_d = row.index("Date")
                row_cp = row.index("Close")
                row_v = row.index("Volume")
                i = True

    close_prices.reverse()
    volumes.reverse()
    dates.reverse()
    return close_prices, volumes, dates