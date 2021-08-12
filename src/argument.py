import argparse
import os


def parse():
    '''
    Add arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Bird-Species-Classification-Using-Transfer-Learning')

    parser.add_argument('--root_dir', type=str,
                        default=os.environ.get('DATA_DIR')+'/images', help='root directory of dataset')
    parser.add_argument('--output_dir', type=str,
                        default=os.environ.get('TRAINING_DIR')+'/checkpoint/', help='directory of saved checkpoints')
    parser.add_argument('--num_epochs', type=int,
                        default=200, help='number of epochs')
    parser.add_argument('--D', type=int,
                        default=6, help='number of dilated convolutional layer')
    parser.add_argument('--C', type=int,
                        default=64, help='kernel size of convolutional layer')
    parser.add_argument('--plot', type=bool, default=False,
                        help='plot loss during training or not')
    parser.add_argument('--model', type=str, default='dudncnn',
                        help='dncnn, udncnn, or dudncnn')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate for training')
    parser.add_argument('--image_size', type=tuple, default=(180, 180))
    parser.add_argument('--test_image_size', type=tuple, default=(320, 320))
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sigma', type=int, default=30)
    parser.add_argument(
        '--quantize',
        choices=[None, 'static', 'fx_static'],
        default=None
    )

    return parser.parse_args()


class Args():
    '''
    For jupyter notebook
    '''

    def __init__(self):
        self.root_dir = os.environ.get('DATA_DIR')+'/images'
        self.output_dir = os.environ.get('TRAINING_DIR')+'/checkpoint/'
        self.num_epochs = 200
        self.D = 6
        self.C = 64
        self.plot = False
        self.model = 'dudncnn'
        self.lr = 1e-3
        self.image_size = (180, 180)
        self.test_image_size = (320, 320)
        self.batch_size = 4
        self.sigma = 30

class ArgsQ():
    '''
    For qtest jupyter notebook
    '''

    def __init__(self):
        self.root_dir = os.environ.get('DATA_DIR')+'/images'
        self.output_dir = os.environ.get('TRAINING_DIR')+'/checkpoint/'
        self.num_epochs = 200
        self.D = 6
        self.C = 64
        self.plot = False
        self.model = 'dudncnn'
        self.lr = 1e-3
        self.image_size = (180, 180)
        self.test_image_size = (320, 320)
        self.batch_size = 4
        self.sigma = 30
        self.image = '../test.jpg'
        self.output = None
        self.quantize = None
        self.show=False

