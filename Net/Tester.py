import torch
import os
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class Tester:
    def __init__(self, model, type, dataloader):
        self.type = type
        self.model = model
        self.dataloader = dataloader

        #find available device
        self.device = None
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        #move network
        self.model.to(self.device)
        
    def load_model(self, path_to_checkpoint):
        #load the model back
        checkpoint = torch.load(path_to_checkpoint)
        self.model.load_state_dict(checkpoint['model'])
    
    def test(self, repetition, log_folder=None):
        #initialize folders
        log_folder = self.initialize_folder('test_log', log_folder)

        #initalize global variables
        inference_time = 0
        writer = SummaryWriter(log_folder)

        ######
        # Credit to Amnon Geifman for the timing method
        # https://towardsdatascience.com/the-correct-way-to-measure-inference-time-of-deep-neural-networks-304a54e5187f
        ######

        #warming process
        print('Warming up...')
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(self.dataloader)):
                data = data.to(self.device)
                _ = self.model(data)

        #testing
        print('Testing...')
        with torch.no_grad():
            for i in range(repetition):
                epoch_time = 0

                for batch_idx, (data, target) in enumerate(tqdm(self.dataloader)):
                    data = data.to(self.device)
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    start_event.record()
                    _ = self.model(data)
                    end_event.record()

                    torch.cuda.synchronize()

                    epoch_time = epoch_time +\
                            start_event.elapsed_time(end_event) / 1000
                
                inference_time = inference_time + epoch_time

                writer.add_scalar('time per epoch', epoch_time)
                writer.add_scalar('total time', epoch_time)

    def get_default_folder(self, type):
        '''
          A bit confusing here...
          The type parameter here should be either 'log' or 'checkpoints'
        '''
        current_time = datetime.datetime.now()
        current_time = current_time.strftime('%Y-%M-%d')

        folder = '{file_type}/{model_type}_{date}'.format(
            file_type = type,
            model_type = self.type,
            date = current_time
        )

        return folder

    def initialize_folder(self, type, folder):
        '''
          A helper method to initialize folders
        '''
        if not folder:
            folder = self.get_default_folder(type)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        
        return folder