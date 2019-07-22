import torch, math
from utils.logger import Logger
from utils.validator import Validator
from utils.utils import ProgressbarWrapper as Progressbar

def init_training(config, model):
  torch.backends.cudnn.benchmark = True # Optimizes cudnn
  model.to(model.device) # model -> CPU/GPU
  # TODO: Can I benchmark on a small tensor to lower GPU memory footprint?

  # Optimizer & Scheduler
  optimizer = torch.optim.Adam(model.parameters(), weight_decay=config.weight_decay, lr=config.start_learning_rate)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.optim_steps/config.lr_step_frequency, eta_min=config.end_learning_rate)

  return scheduler, optimizer

def train(model, config, dataloader):
  # Init training helpers
  logger = Logger(config)
  validator = Validator(config, dataloader, model, logger)
  lr_scheduler, optimizer = init_training(config, model)

  # Init progressbar
  n_batches = len(dataloader.get_train())
  n_epochs = math.ceil(config.optim_steps / n_batches)
  pbar = Progressbar(n_epochs, n_batches)

  # Init variables
  val_freq = config.validation_frequency
  optim_steps = 0

  # Training loop starts here
  for epoch in pbar(range(1, n_epochs + 1)):
    for batch_i, data in enumerate(dataloader.get_train(), 1):
      pbar.update(epoch, batch_i)

      # End training after a set amount of steps
      if optim_steps >= config.optim_steps:
        break

      # Decrease learning rate
      if optim_steps % config.lr_step_frequency == 0:
        lr_scheduler.step()

      # Validation
      if optim_steps % val_freq == val_freq - 1:
        validator.validate(optim_steps, epoch)

      inputs, labels, _ = data
      image_size = inputs.size(2)
      optimizer.zero_grad()
      outputs = model(inputs)

      losses = model.calc_loss(outputs, labels, image_size)
      sum_loss = sum(losses.values())
      sum_loss.backward()
      optimizer.step()
      optim_steps += 1

      # Frees up GPU memory
      del data; del outputs