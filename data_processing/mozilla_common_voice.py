import pandas as pd
import numpy as np
import os

np.random.seed(999)

class MozillaCommonVoiceDataset:

    def __init__(self, basepath, *, val_dataset_size):
        self.basepath = basepath
        self.val_dataset_size = val_dataset_size

    def _get_common_voice_filenames(self, dataframe_name='train.tsv'):
        mozilla_metadata = pd.read_csv(os.path.join(self.basepath, dataframe_name), sep='\t')
        clean_files = mozilla_metadata['path'].values
        np.random.shuffle(clean_files)
        print("Total number of training examples:", len(clean_files))
        return clean_files

    def get_train_val_filenames(self):
        clean_files = self._get_common_voice_filenames(dataframe_name='train.tsv')

        # resolve full path
        clean_files = [os.path.join(self.basepath, 'clips', 'train', filename) for filename in clean_files]

        clean_files = clean_files[:-self.val_dataset_size]
        clean_val_files = clean_files[-self.val_dataset_size:]
        print("# of Training clean files:", len(clean_files))
        print("# of  Validation clean files:", len(clean_val_files))
        return clean_files, clean_val_files


    def get_test_filenames(self):
        clean_files = self._get_common_voice_filenames(dataframe_name='test.tsv')

        # resolve full path
        clean_files = [os.path.join(self.basepath, 'clips', 'test', filename) for filename in clean_files]

        print("# of Testing clean files:", len(clean_files))
        return clean_files