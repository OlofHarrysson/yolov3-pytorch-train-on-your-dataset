import pyjokes, random
from datetime import datetime as dtime
from collections import OrderedDict

class DefaultConfig():
  def __init__(self, config_str):
    # ~~~~~~~~~~~~~~ General Parameters ~~~~~~~~~~~~~~
    # An optional comment to differentiate this run from others
    self.save_comment = pyjokes.get_joke()
    print('\n{}\n'.format(self.save_comment))

    # Start time to keep track of when the experiment was run
    self.start_time = dtime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Seed to create reproducable training results
    self.seed = random.randint(0, 2**32 - 1)

    # Use a subset of the dataset
    self.sample = True

    # How many samples to group up as input
    self.batch_size = 16

    # Path to weights to load. Set to False to skip loading
    self.weights = 'saved/weights/coco.weights'

    # Freezes different layers. Can freeze None, Darknet 53 module or everything except last layers.
    # Less memory & faster training if more weights are frozen.
    self.weight_freeze = 'dn53' # None, 'dn53' or 'all_but_last' 

    # Skip loading last weight layers. Do this if you have different amount of classes than the weights
    self.skip_last = False

    # Use GPU. Set to False to only use CPU
    self.use_gpu = True

    # The config name
    self.config = config_str

    # The folder name of the dataset
    self.datadir = self.config

    # The folder name where the classes.json is
    self.classdir = self.config

    # Number of threads to use when reading data
    self.data_read_threads = 3

    # How strong the objectness prediction needs to be for it to be considered a positive prediction. Used to filter before non-maximal supression
    self.confidence_thresh = 0.25

    # The IoU overlap needed to supress the weaker prediction
    self.non_maximal_supression_thresh = 0.1

    # The IoU threshhold a detection must achive for the detection to count as a correct prediction
    self.iou_thresh = 0.5

    # ~~~~~~~~~~~~~~ Training Specific Parameters ~~~~~~~~~~~~~~
    # The maximum of optimization steps before training stops
    self.optim_steps = 75000

    # Learning rate
    self.start_learning_rate = 1e-3

    # Decays learning rate with cosine annealing
    self.end_learning_rate = 1e-4

    # Decreases the learning rate every x'th step
    self.lr_step_frequency = 1000

    # Which datasets to validate on
    self.validations = []

    # How many optimization steps between validation
    self.validation_frequency = 1000

    # Max number of validation batches for any single validation. Avoids spending too much time on validating
    self.max_validation_batches = 50

    # Which dataset to save the best weights for. 
    self.weight_save_data = ''

    # Save the weights when a metric improves. Supports multiple metrics and saves them at different paths.
    # Any combination of: 'ap', 'f1', 'precision', 'recall', 'loss'
    self.weight_save_criteria = ['ap', 'f1', 'loss']

    # Size the input image is resized to during evaluation mode
    self.eval_image_sizes = [416]

    # Multiscaling. Samples a random image size for each epoch
    self.train_image_sizes = [352, 384, 416, 448, 480, 512, 544]

    # Loss lambdas. Multipliers of the different parts of the loss function. A higher multiplier will increase that part of the loss function making it more important
    self.lambda_x = 1
    self.lambda_y = 1
    self.lambda_w = 1
    self.lambda_h = 1
    self.lambda_obj_gt = 1 # Cells with ground truth
    self.lambda_obj_no_gt = 1 # Cells without any object
    self.lambda_class = 1

    # Weight decay
    self.weight_decay = 5e-4

  def get_parameters(self):
    return OrderedDict(sorted(vars(self).items()))

  def __str__(self): # TODO return str
    # class name, etc
    return str(vars(self))

class NoCuda(DefaultConfig):
  def __init__(self, config_str):
    super().__init__(config_str)
    self.use_gpu = False

class Coco(DefaultConfig):
  def __init__(self, config_str):
    super().__init__(config_str)
    ''' Change default parameters here. Like this
    self.seed = 666          ____
      ________________________/ O  \___/  <--- Python <3
     <_#_#_#_#_#_#_#_#_#_#_#_#_____/   \
    '''

class Faces(DefaultConfig):
  def __init__(self, config_str):
    super().__init__(config_str)
    self.validations = ['faces_train', 'faces_val']
    self.batch_size = 8
    self.optim_steps = 1000
    self.validation_frequency = 20
    self.skip_last = True
    self.weights = 'saved/weights/one_class.weights'
    self.weight_freeze = 'dn53'
    self.train_image_sizes = [416]
    self.data_read_threads = 0

class Fruits(DefaultConfig):
  def __init__(self, config_str):
    super().__init__(config_str)
    self.validations = ['fruits_train', 'fruits_val']
    self.optim_steps = 1e4
    self.validation_frequency = 1
    # self.validation_frequency = 50
    self.skip_last = True
    self.batch_size = 8
    self.data_read_threads = 0