import csv

sub_file="training.csv"
x_min=1000
x_max=0
y_min=1000
y_max=0
with open(sub_file) as f:
    csv_reader = csv.reader(f,delimiter=',')
    line=0
    for x in csv_reader:
        if line==0 :
            line=1
            continue
        x[1],x[2],x[3],x[4]=float(x[1]),float(x[2]),float(x[3]),float(x[4])
        if(x[1]<x_min):
             x_min=x[1]
        if(x[2]>x_max):
             x_max=x[2]
        if(x[3]<y_min):
             y_min=x[3]
        if(x[4]>y_max):
             y_max=x[4]
print(x_min," --x-- ",x_max)
print(y_min," --x-- ",y_max)
