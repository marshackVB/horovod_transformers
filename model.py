import pytorch_lightning as pl
from transformers import AutoConfig, AutoModelForSequenceClassification, PretrainedConfig
from torch.optim import AdamW


class TransformerModel(pl.LightningModule):
    def __init__(self, model_name:str, model_config:PretrainedConfig, lr=5e-5, device_id=0, device_count=1):
        super().__init__()
        self.model_name = model_name
        self.model_config = model_config
        self.lr = lr
        self.device_id = device_id
        self.device_count = device_count
        self.model = self.init_model()
        
        
    def init_model(self):
      return AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.model_config)
        
      
    def forward(self, x):
      output = self.model(**x)
      return output
      
      
    def configure_optimizers(self):
      optimizer = AdamW(self.parameters(), lr=self.lr)
      return optimizer
    
    
    def training_step(self, train_batch, batch_idx):
      output = self.model(**train_batch)
      loss = output.loss
      self.log('train_loss', loss)
      return loss
      
      
    def validation_step(self, val_batch, batch_idx):
      output = self.model(**val_batch)
      loss = output.loss
      self.log("val_loss", loss)
      
    #def backward(trainer, loss, optimizer, optimizer_idx):
    #  loss.backward()
      
    #def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,):
    #  optimizer.step()
      
      
      
      
      
      