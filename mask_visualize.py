import os
import numpy as np
import matplotlib.pyplot as plt


class draw_mask(object):
    def __init__(self, folder="/home/liao/work_code/LostGANs/samples/tmp/edit"):
        self.folder_path = os.path.abspath(folder)

    def files_in_folder(self, mypath, extension=''):
        file_list = []
        for root, dirs, files in os.walk(mypath):
            for file in files:
                if file.endswith(extension) and ('_pic.' not in file):
                    file_path = os.path.join(root, file)
                    file_list.append(file_path)
        return file_list

    def plot_save_numpy_arrya(self, data, export_path=''):
        if export_path:
            plt.imsave(export_path, data)
            print("-----> Fig has been saved as: {}\n".format(export_path))

    def run(self):
        data = []
        file_list = self.files_in_folder(self.folder_path, '.npy')
        for f_path in file_list:
            print("Analyzing file {}.....".format(f_path))
            data = np.load(f_path)
            # print(data.shape)
            data = np.squeeze(data)
            mask = np.argmax(data[:, :, :], axis=0)
            f1 = os.path.dirname(f_path)
            f2 = os.path.basename(f_path).split('.')[0] + '_pic.png'
            pic_path = os.path.join(f1, f2)
            self.plot_save_numpy_arrya(mask, pic_path)


# test = draw_mask(r'C:\Users\jutang\Documents\Work\Python\wt_cvpr\app')
# test.run()
