import pytorch_lightning as pl

class TransformersDataModule(pl.LightningDataModule):
  def __init__(self, train_converter, val_converter, batch_size, device_id:int=0, device_count:int=1):
    self.train_converter = train_converter
    self.val_converter = val_converter
    self.batch_size = batch_size
    self.device_id = device_id
    self.device_count = device_count
    self.train_dataloader_context = None
    self.val_dataloader_context = None
    self.prepare_data_per_node = False
    self._log_hyperparams = False
    
    
  def train_dataloader(self):
    self.train_dataloader_context = self.train_converter.make_torch_dataloader(num_epochs=None,
                                                                               cur_shard=self.device_id,
                                                                               shard_count=self.device_count,
                                                                               batch_size=self.batch_size * self.device_count)
    return self.train_dataloader_context.__enter__()
    #return iter(self.train_dataloader_context)
  
  
  def val_dataloader(self):
    self.val_dataloader_context = self.val_converter.make_torch_dataloader(num_epochs=None,
                                                                           cur_shard=self.device_id,
                                                                           shard_count=self.device_count,
                                                                           batch_size=self.batch_size * self.device_count)
    return self.val_dataloader_context.__enter__()
    #return iter(self.val_dataloader_context)
    
    
  def teardown(self, stage=None):
    self.train_dataloader_context.__exit__(None, None, None)
    self.val_dataloader_context.__exit__(None, None, None)

    
    

    
   
  
  
  