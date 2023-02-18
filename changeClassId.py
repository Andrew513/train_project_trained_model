import os
all_files = os.listdir("C:/Users/DilDoooo/Desktop/cproject/yolov5_3class/valid/labels")
def inplace_change(filename, old_string, new_string):
    # Safely read the input filename using 'with'
    with open(filename) as f:
        s = f.read()
        if old_string not in s:
            print('"{old_string}" not found in {filename}.'.format(**locals()))
            return

    # Safely write the changed content, if found in the file
    with open(filename, 'w') as f:
        print('Changing "{old_string}" to "{new_string}" in {filename}'.format(**locals()))
        s = s.replace(old_string, new_string)
        f.write(s)
#print(all_files)
for file in all_files:
    inplace_change(file,"1 ","80 ")
