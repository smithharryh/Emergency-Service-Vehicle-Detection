import os
import wavio
import shutil

for file_name in os.listdir('Data/Emergency/'):
    print("round")
    try:
        wavio.read('Data/Emergency/' +file_name)
    except:
        print(file_name)
        shutil.move("Data/Emergency/"+file_name, "../brokens")

