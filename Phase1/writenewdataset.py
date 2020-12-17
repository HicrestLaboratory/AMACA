fileName=input("Name of dataset: ")
print("Creating file: "+fileName+".csv")
file=open(fileName+".csv","w")
file.write(",W,H,C_IN,KERNEL_SIZE,C_OUT (filters count),STRIDE,PAD,,Layer name\n")

read_params = ['KSIZE','PAD','STRIDE','CIN','W','COUT']
params = ['W','W','CIN','KSIZE','COUT','STRIDE','PAD']


while True:
    if input("Enter to continue. Everything else to exit") != '':
        break
    value = {};
    try:
        for p in read_params:
            value[p] = int(input(p + ': '));
        string = fileName;
        for p in params:
            string += ',' + str(value[p]);

        string += ',,test \n'
        if input(string) != '':
            print("linea annullata")
            continue
        file.write(string);
    except:
        print("linea annullata")
        continue


print("Generated "+ fileName +" tests")
file.close()
