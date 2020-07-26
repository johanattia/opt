import os
import tqdm
import numpy as np
import tensorflow as tf


class Trainer(object):
    
    def __init__(self, model, optimizer, loss_object, early_stopping=False, patience=None, restore_best_model=False, checkpoint_directory=None):
        
        self.model = model
        self.optimizer = optimizer
        self.loss_object = loss_object
        
        self.early_stopping = early_stopping
        self.restore_best_model = restore_best_model
        
        if self.early_stopping or self.restore_best_model:
            
            if checkpoint_directory is not None and os.path.isdir(checkpoint_directory):
                self.checkpoint = tf.train.Checkpoint(model=self.model)
                self.checkpoint_manager = tf.train.CheckpointManager(
                    self.checkpoint, 
                    directory=checkpoint_directory, 
                    max_to_keep=3
                )
                
            else :
                raise ValueError('A valid directory must be filled for checkpoint_directory.')
                
            self.best_epoch = 1
            self.best_val_loss = np.inf
            self.finally_restored = False
           
        if self.early_stopping:
            
            try:
                self.patience = int(patience)
                
            except:
                raise Exception('Patience must be an integer value.')
                
        self.history = {'train_loss': [], 'val_loss': []}
        
    def train(self, train_dataset, validation_dataset, epochs, mode='auto-encoder'):
        
        for epoch in tqdm.notebook.tqdm(range(1, epochs+1), desc = 'T R A I N I N G') :

            train_loss = tf.keras.metrics.Mean()
            val_loss = tf.keras.metrics.Mean()

            for data in train_dataset: 
                self.train_step(data, train_loss, mode=mode)

            for data in validation_dataset:
                self.validation_step(data, val_loss, mode=mode)
            
            if (self.restore_best_model or self.early_stopping) and (val_loss.result().numpy() < self.best_val_loss):
                self.best_epoch = epoch
                self.best_val_loss = val_loss.result().numpy()
                self.checkpoint_manager.save()
                
            print (f"[Epoch {epoch}/{epochs}] train_loss: {train_loss.result()} - val_loss: {val_loss.result()}")
            
            self.history['train_loss'].append(train_loss.result().numpy())
            self.history['val_loss'].append(val_loss.result().numpy())

            if self.early_stopping and (epoch - self.best_epoch > self.patience):
                self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
                self.finally_restored = True

                print(f"Patience exceeded : training done ! Best epoch : {self.best_epoch}. Best model restored.")
                break
                
        #if self.restore_best_model and (epoch != self.best_epoch): # and not self.finally_restored:
        #    if self.finally_restored:
        #        pass
        #    
        #    else:
        #        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        #        self.finally_restored = True
        #        print(f"Training done ! Best epoch : {self.best_epoch}. Best model restored.")
        
        return
    
    def train_step(self, data, train_loss, mode):
        
        inputs, labels = data
        
        if mode == 'auto-encoder':
            labels = inputs

        with tf.GradientTape() as tape: 
            outputs = self.model(inputs)
            loss = self.loss_object(labels, outputs)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        train_loss.update_state(loss)
        
        return
        
    def validation_step(self, data, val_loss, mode):
        
        inputs, labels = data
        
        if mode == 'auto-encoder':
            labels = inputs

        outputs = self.model(inputs)
        loss = self.loss_object(labels, outputs)
        val_loss.update_state(loss)
        
        return