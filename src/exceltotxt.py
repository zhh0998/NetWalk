import csv

with open("./tmp/citeseer_label.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    outF = open("./tmp/membership_citeseer.txt", "w")
    for row in csv_reader:
        outF.write(str(row[0]))
        outF.write("\n")
    outF.close()