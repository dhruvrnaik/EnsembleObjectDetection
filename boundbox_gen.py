import cv2
import csv
csv_file="training_set.csv"
with open(csv_file) as f:
	csv_reader = csv.reader(f, delimiter=',')
	next(csv_reader)
	for row in csv_reader:
		path = "training/"+row[0]
		img = cv2.imread(path,1)
		cv2.rectangle(img,(int(float(row[1])),int(float(row[3]))),(int(float(row[2])),int(float(row[4]))),(255,255,0),2)
		cv2.imwrite("trainbox_actual/"+row[0],img)
