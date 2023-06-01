import torch
import os
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, type, dataloader, epoch, lr=0.001):
        #initialize all variables
        self.type = type
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()
        self.dataloader = dataloader
        self.current_epoch = 0
        self.target_epoch = epoch

        #find available device
        self.device = None
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        #move network
        self.model.to(self.device)
        self.loss.to(self.device)

        #initialize optimizer (as recommened by pytorch)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def resume(self, path_to_checkpoint):
        #load the model and optimizer back
        checkpoint = torch.load(path_to_checkpoint)

        self.current_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train(self, save_per_epoch, validation=0.1, early_end=3,\
              checkpoint_folder=None, max_checkpoints=5,\
              log_folder=None):
        #initialize folders
        checkpoint_folder = self.initialize_folder('checkpoint', checkpoint_folder)
        log_folder = self.initialize_folder('log', log_folder)

        #initalize global variables
        no_improvement_count = 0
        validation_start_idx = len(self.dataloader) * (1-validation)
        best_val_score = 0
        train_time = 0
        writer = SummaryWriter(log_folder)

        #output graph if needed
        if self.current_epoch == 0:
            data, target = next(iter(self.dataloader))
            data = data.to(self.device)
            writer.add_graph(self.model, data)

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

        #training process
        print('Start training...')
        while self.current_epoch < self.target_epoch:
            #initialize step, loss list (to compute avg epoch loss)
            #and val list (for validation)
            epoch_time = 0
            loss_list = []
            val_list = []

            for batch_idx, (data, target) in enumerate(tqdm(self.dataloader)):
                #move data and target
                data = data.to(self.device)
                target = target.to(self.device)

                if batch_idx < validation_start_idx:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    #start timer
                    start_event.record()

                    #trainning
                    self.optimizer.zero_grad()
                    out = self.model(data)
                    loss = self.loss(out, target)
                    loss.backward()
                    self.optimizer.step()

                    #end timer
                    end_event.record()

                    #synchronize process
                    torch.cuda.synchronize()

                    #record time
                    epoch_time = epoch_time +\
                            start_event.elapsed_time(end_event) / 1000

                    #append loss
                    loss_list.append(loss.detach().cpu().item())

                    #clear cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                else:
                    #validation
                    with torch.no_grad():
                        out = self.model(data)
                        out = torch.argmax(out, dim=1)
                        target = torch.argmax(target, dim=1)
                        val_list.append(out==target)
            
            #calculate avg loss
            avg_loss = sum(loss_list) / len(loss_list)

            #calculate val score
            temp = torch.cat(val_list)
            val_score = torch.sum(temp) / temp.shape[0]
            val_score = val_score.detach().cpu().item()

            #update trainning time
            train_time = train_time + epoch_time

            #print information
            print('Epoch: {epoch}, Loss: {curr_loss}, Val: {curr_val}%'.format(
                        epoch = self.current_epoch,
                        curr_loss = avg_loss,
                        curr_val = round(100*val_score, 2)
                ))

            #compare val score
            if val_score > best_val_score:
                #update best val score
                no_improvement_count = 0
                best_val_score = val_score
            else:
                #update patience
                no_improvement_count = no_improvement_count + 1

                if no_improvement_count > early_end:
                    self.save_checkpoint(checkpoint_folder, max_checkpoints,\
                                         round(avg_loss, 4))
                    print('Early termination.')
                    break

            #save model if needed
            if (self.current_epoch + 1) % save_per_epoch == 0:
                self.save_checkpoint(checkpoint_folder, max_checkpoints,\
                                     round(avg_loss, 4))

            #log values
            writer.add_scalar('loss', avg_loss, self.current_epoch)
            writer.add_scalar('validation', 100*val_score, self.current_epoch)
            writer.add_scalar('training time per epoch',\
                                epoch_time, self.current_epoch)
            writer.add_scalar('total trainning time',\
                                train_time, self.current_epoch)

            #finalize an epoch
            self.current_epoch = self.current_epoch + 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

    def save_checkpoint(self, checkpoint_folder, max_checkpoints, loss):
        #generate model name
        model_name = '{type}_epoch{curr_epoch}_loss{curr_loss}.checkpoint'\
                                    .format(type = self.type,\
                                            curr_epoch = self.current_epoch,\
                                            curr_loss = loss)
        #generate checkpoint
        checkpoint = {
            'epoch': self.current_epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        #save checkpoint
        torch.save(checkpoint, os.path.join(checkpoint_folder, model_name))

        #check model count
        file_list = os.listdir(checkpoint_folder)
        if len(file_list) > max_checkpoints:
            #remove the earliest one
            file_list.sort(key = lambda x: os.path.getmtime(
                os.path.join(checkpoint_folder, x)
            ))
            os.remove(os.path.join(checkpoint_folder, file_list[0]))