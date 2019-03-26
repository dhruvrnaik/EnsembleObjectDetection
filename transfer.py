import shutil
import csv
with open("xml/new_vals.csv") as f:
	csv_reader = csv.reader(f)
	for x in csv_reader:
		shutil.copy('images/'+x[0],"aug_new_train/"+x[0])
