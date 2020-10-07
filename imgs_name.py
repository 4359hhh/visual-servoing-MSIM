import os
import numpy as np

images_path = './process/'
imgs_name = []

with open('imgs_name_process.txt','w') as f:
    for filename in os.listdir(images_path):

        path = images_path+filename
        imgs_name.append(path)
        print(path)
    #np.save('imgs_name.txt',imgs_name)
        f.writelines(path+'\n')


###a bug on ubantu ,don't run!!!







