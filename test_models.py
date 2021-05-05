import unittest
from models import Conv1D, Conv2D, LSTM
from CleanAndSplit import split_wavs
from Website import make_predictions
import argparse
import os
from tqdm import tqdm
import wavio

class TestModelBuilding(unittest.TestCase):
    def test_classifier(self):
        # Test that the code works correclty when assigned a number of classes.
        number_of_classes = 3
        model = Conv2D(NUMBER_CLASSES=number_of_classes)
        model_shape = (model.get_layer('dense_1').output_shape)
        self.assertEqual((None, number_of_classes),model_shape)

    def test_input_shape(self):
        # Test the input shape calculation works when altering sample rate and delta time
        sample_rate = 15000
        delta_time = 1.0
        model_input_shape = LSTM(SAMPLE_RATE=sample_rate).input_shape
        self.assertEqual(model_input_shape, (None, sample_rate, int(delta_time)))


class TestDataCleaning(unittest.TestCase):
    def test_number_of_classes(self):
        # Test to see if number of classes specified in parser is used in all places.
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', type=str, default='testData'), 
        parser.add_argument('--dst_root', type=str, default='testClean'),
        parser.add_argument('--delta_time', type=float, default=1.0)
        parser.add_argument('--sample_rate', type=int, default=16000)
        parser.add_argument('--threshold', type=str,default=100)
        args, _ = parser.parse_known_args()
        input_classes = 2
        output_classes = os.listdir(args.dst_root).__len__()
        class_num = split_wavs(args)
        self.assertEqual(input_classes,output_classes)  
    
    def test_downsample_function(self):        
        classes = os.listdir('testClean')
        for _class in classes:
            if(_class != '.DS_Store'):
                target_dir = os.path.join('testClean', _class)
                for fn in os.listdir(target_dir):
                   if(fn != '.DS_Store'):
                        wav = wavio.read(os.path.join(target_dir, fn))
                        self.assertEqual(wav.rate, 16000)

class TestWebsite(unittest.TestCase):
    def test_prediction(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_fn', type=str, default='models/LSTM.h5')
        parser.add_argument('--delta_time', type=float, default=1.0)
        parser.add_argument('--sample_rate', type=int, default=16000)
        parser.add_argument('--threshold', type=str, default=20)
        # parser.add_argument('--src_dir', type=str, default='TestData')

        args, _ = parser.parse_known_args()
        test_file = 'testData/Emergency/10.wav'

        prediction =  make_predictions(args, test_file)
        self.assertEqual(prediction, 'emergency')