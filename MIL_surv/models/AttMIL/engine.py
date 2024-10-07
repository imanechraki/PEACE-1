import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from sksurv.metrics import concordance_index_censored
import wandb 
import torch.optim


class Engine(object):
    def __init__(self, args, results_dir, fold):
        self.args = args
        self.results_dir = results_dir
        self.fold = fold
        self.delta_results_path = os.path.join(results_dir, 'fold_' + str(fold)+'.csv')
        # tensorboard
        if args.log_data:
            from tensorboardX import SummaryWriter
            writer_dir = os.path.join(results_dir, 'fold_' + str(fold))
            self.writer = SummaryWriter(writer_dir, flush_secs=15)

        self.best_scores = 0
        self.best_epoch = 0
        self.delta_scores = None
        self.data_IDs = None
        self.filename_best = None

    def learning(self, model, train_loader, val_loader, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()
        if self.args.resume is not None:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.best_scores = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint (score: {})".format(checkpoint['best_score']))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))

        if self.args.evaluate:
            self.validate(val_loader, model, criterion)
            return

        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            # train for one epoch
            self.train(train_loader, model, criterion, optimizer)
            # evaluate on validation set
            scores, ID_delta_scores = self.validate(val_loader, model, criterion)
            # remember best c-index and save checkpoint
            is_best = scores > self.best_scores
            if is_best:
                self.data_IDs, self.delta_scores = ID_delta_scores
                df_results = pd.DataFrame({'ID':self.data_IDs,'delta_score':self.delta_scores})
                self.best_scores = scores
                self.best_epoch = self.epoch
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_score': self.best_scores})
            print(' *** best score={:.4f} at epoch {}'.format(self.best_scores, self.best_epoch))
            scheduler.step()
            print('>>>')
            print('>>>')
        df_results.to_csv(self.delta_results_path)
        return self.best_scores, self.best_epoch

    def train(self, data_loader, model, criterion, optimizer):
        model.train()

        total_loss = 0.0
        all_delta_scores = np.zeros((len(data_loader)))
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc='Train Epoch {}'.format(self.epoch))
        for batch_idx, (data_ID, data_WSI,data_clinical,RX_f, RX_cf, data_Event, data_Censorship, data_Label, delta) in enumerate(dataloader):
            if torch.cuda.is_available():                
                data_WSI = data_WSI.cuda()
                data_clinical = data_clinical.cuda()

                data_Label = data_Label.type(torch.LongTensor).cuda()
                data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
                RX_f = RX_f.cuda()
                RX_cf = RX_cf.cuda()
                delta = delta.cuda()
                data = {'wsi':data_WSI,'clinical':data_clinical, 'RX_f': RX_f, 'RX_cf':RX_cf}
                
            # prediction
            hazards, S, delta_pred, risks = model(data)

            loss = criterion(hazards=hazards, S=S, Y=data_Label, c=data_Censorship, delta = delta, delta_pred = delta_pred)

            # results
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_delta_scores[batch_idx] = delta_pred.detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = data_Event
            total_loss += loss.item()
            # backward to update parameters
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # calculate loss and error for each epoch
        loss = total_loss / len(dataloader)
        try:
            c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        except:
            print(all_censorships)
            print(all_risk_scores)
            print(all_event_times)
        print('loss: {:.4f}, c_index: {:.4f}'.format(loss, c_index))
        if self.writer:
            self.writer.add_scalar('train/loss', loss, self.epoch)
            self.writer.add_scalar('train/c_index', c_index, self.epoch)

      
            wandb.log({
            "epoch": self.epoch,
            "train_loss": loss,
            "train_ci": c_index,
        })
    


    def validate(self, data_loader, model, criterion):
        model.eval()
        total_loss = 0.0
        all_data_ID = np.zeros((len(data_loader)))
        all_delta_scores = np.zeros((len(data_loader)))
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc='Test Epoch {}'.format(self.epoch))

        for batch_idx, (data_ID, data_WSI, data_clinical,RX_f, RX_cf, data_Event, data_Censorship, data_Label, delta) in enumerate(dataloader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                data_clinical = data_clinical.cuda()   
                RX_f = RX_f.cuda()
                RX_cf = RX_cf.cuda()
                delta = delta.cuda()
                data = {'wsi':data_WSI,'clinical':data_clinical, 'RX_f': RX_f, 'RX_cf':RX_cf}    
                data_Label = data_Label.type(torch.LongTensor).cuda()
                data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
            # prediction
            with torch.no_grad():
                hazards, S, delta_pred, risks = model(data)
            loss = criterion(hazards=hazards, S=S, Y=data_Label, c=data_Censorship, delta= delta, delta_pred = delta_pred)
            total_loss += loss.item()
            # results
            
            all_delta_scores[batch_idx] = delta_pred.detach().cpu().numpy()
            all_data_ID[batch_idx] = data_ID.item()

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = data_Event
            
        # calculate loss and error for each epoch
        loss = total_loss / len(dataloader)
        # c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        try:
            c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        except:
            print(all_censorships)
            print(all_risk_scores)
            print(all_event_times)
        print('loss: {:.4f}, c_index: {:.4f}'.format(loss, c_index))
        if self.writer:
            self.writer.add_scalar('val/loss', loss, self.epoch)
            self.writer.add_scalar('val/c_index', c_index, self.epoch)
            
            wandb.log({
            "epoch": self.epoch,
            "val_loss": loss,
            "val_ci": c_index,
        })

        

        return c_index, (all_data_ID,all_delta_scores)

    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.results_dir,
                                          'fold_' + str(self.fold),
                                          'model_best_{score:.4f}_{epoch}.pth.tar'.format(score=state['best_score'], epoch=state['epoch']))
        print('save best model {filename}'.format(filename=self.filename_best))
        torch.save(state, self.filename_best)
