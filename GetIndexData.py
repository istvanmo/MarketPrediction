import csv
import matplotlib.pyplot as plt

def get_data():
    close_prices = []
    volumes = []
    dates = []
    with open('N225.csv', newline='') as csvfile:
        N225 = csv.reader(csvfile)
        i = False
        row_d = None
        row_cp = None
        row_v = None
        n = []
        for r in N225:
            if i:
                n.append(r)
            else:
                row_d = r.index("Date")
                row_cp = r.index("Close")
                row_v = r.index("Volume")
                i = True

        for row in n:
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

    print("close prices: ", close_prices[-15:])
    print("volumes: ", volumes[-15:])
    print("dates: ", dates[-15:])

    return close_prices, volumes, dates