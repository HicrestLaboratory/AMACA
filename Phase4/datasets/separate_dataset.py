precision = 'uint8';

fileARFF	="dataset.arff"
newfile = "dataset_" + precision + ".arff";

purgedARFF	=open(newfile,"w")

if precision == 'default':
    precision = 0;
else:
    precision = 1;

purgedARFF.write("@RELATION convolution\n")
purgedARFF.write("@ATTRIBUTE tensor_1 INTEGER\n")
purgedARFF.write("@ATTRIBUTE tensor_2 INTEGER\n")
purgedARFF.write("@ATTRIBUTE tensor_3 INTEGER\n")
purgedARFF.write("@ATTRIBUTE tensor_4 INTEGER\n")
purgedARFF.write("@ATTRIBUTE tensor_5 INTEGER\n")
purgedARFF.write("@ATTRIBUTE tensor_6 INTEGER\n")
purgedARFF.write("@ATTRIBUTE tensor_7 INTEGER\n")
purgedARFF.write("@ATTRIBUTE tuner INTEGER\n")
purgedARFF.write("@ATTRIBUTE precision INTEGER\n")
purgedARFF.write("@ATTRIBUTE architecture INTEGER\n")
purgedARFF.write("@ATTRIBUTE class {conv,directconv,winogradconv}\n")
purgedARFF.write("@DATA\n")
arffLines = [line.rstrip('\n') for line in open(fileARFF)]
for i in range(1,len(arffLines)):
    
    arffelement	=arffLines[i].split(",")	
    
    if(len(arffelement)<4):
        continue;
    
    if(arffelement[-3] != str(precision)):
        print(arffelement);
        continue;
    purgedARFF.write(arffLines[i]+"\n")
    
    
    
print("result file: " + newfile)
print("Done")