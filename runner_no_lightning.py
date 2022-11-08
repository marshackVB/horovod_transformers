# Databricks notebook source
# MAGIC %load_ext autoreload 
# MAGIC %autoreload 2

# COMMAND ----------

import shutil
import horovod.torch as hvd
from sparkdl import HorovodRunner

import numpy as np

from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.sql.types import  StructType, StructField, IntegerType, ArrayType
from pyspark.sql.functions import col, pandas_udf
import pyspark.sql.functions as func

import torch
from torch.optim import AdamW

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

from tokenizer import TableTokenizer

# COMMAND ----------

# MAGIC %md Tokenize and convert using Petastorm

# COMMAND ----------

MODEL_NAME = "distilbert-base-uncased"

config = {"model_name": MODEL_NAME,
          "batch_size": 1000,
          "truncation": True,
          "padding": "max_length",
          "max_length": 200}

tokenizer = TableTokenizer(**config)

train_df = tokenizer.tokenize_table('imdb_train', 'text')
test_df = tokenizer.tokenize_table('imdb_test', 'text')

# Set a cache directory for Petastorm
petastorm_cache_dir = 'dbfs:/tmp/petastorm/cache'
dbutils.fs.mkdirs(petastorm_cache_dir)

python_format_dir = f"file:///{petastorm_cache_dir.replace(':', '')}"

train_cache_dir = f"{python_format_dir }/train/"
test_cache_dir = f"{python_format_dir }/test/"

gpu_cnt = 2

spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, train_cache_dir)
converter_train = make_spark_converter(train_df.repartition(gpu_cnt))
               
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, test_cache_dir)
converter_test = make_spark_converter(test_df.repartition(gpu_cnt))

# COMMAND ----------

# MAGIC %md Initialize model

# COMMAND ----------

labels_table_name = 'imdb_labels'
labels = spark.table(labels_table_name)
collected_labels = labels.collect()
 
id2label = {row.idx: row.label for row in collected_labels} 
label2id = {row.label: row.idx for row in collected_labels}

def model_init(model_type_or_path=MODEL_NAME, id2label=id2label, label2id=label2id, 
               problem_type="single_label_classification", num_labels=2):
  
  model_config = AutoConfig.from_pretrained(model_type_or_path, 
                                            num_labels=num_labels,
                                            id2label=id2label, 
                                            label2id=label2id,
                                            problem_type=problem_type)
  
  return AutoModelForSequenceClassification.from_pretrained(model_type_or_path, 
                                                            config=model_config)

# COMMAND ----------

def train_one_epoch(model, dataloader, optimizer, device, steps_per_epoch, logging_steps, epoch, epochs):
  """A single epoch training loop
  """
  losses = []
  model.train
  for step in range(steps_per_epoch):
    
    batch = next(dataloader)
    batch = {k: v.to(device) for k, v in batch.items()}
    
    optimizer.zero_grad()
    
    output = model(**batch)
    
    loss = output.loss
    losses.append(loss.item())
    loss.backward()
    
    optimizer.step()
    
    if step % logging_steps == 0:
      step_loss = torch.tensor(loss.item())
      all_node_avg_step_loss = hvd.allreduce(step_loss, name='avg_loss')
      
      if hvd.rank() == 0:
        if step == 0:
           print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Step {step + 1}/{steps_per_epoch} mean loss: {round(all_node_avg_step_loss.item(), 4)}")
      
  epoch_mean_loss = torch.tensor(np.mean(losses))
  all_node_epoch_mean_loss = hvd.allreduce(epoch_mean_loss, name="epoch_mean_loss")
    
  if hvd.rank() == 0:
    print(f"Epoch {epoch + 1} mean loss: {round(all_node_epoch_mean_loss.item(), 4)}")
      
    
def evaluate(model, dataloader, device, steps_per_epoch):
  
  if hvd.rank() == 0:
      print(f"Running evaluation...")
  
  correct_preds_cnt = 0
  total_preds_cnt = 0
      
  model.eval()
  for step in range(steps_per_epoch):
    batch = next(dataloader)
    batch = {k: v.to(device) for k, v in batch.items()}
    
    with torch.no_grad():
      output = model(**batch)
    
      probabilities = torch.nn.functional.softmax(output.logits, dim=1)
      predicted_classes = probabilities.argmax(dim=-1)
      correct_predictions = torch.eq(predicted_classes, batch["labels"]).sum().item()

      total_preds_cnt += predicted_classes.size(dim=0)
      correct_preds_cnt += correct_predictions
    
  return (correct_preds_cnt, total_preds_cnt)

# COMMAND ----------

def main(model_save_dir, epochs=3, train_batch_size=32, eval_batch_size=64, backward_passes_per_step=4, logging_steps=50):
  
  # horovod init and device configuration
  hvd.init()
  total_number_of_gpus = hvd.size()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if device.type == 'cuda':
    # Pin each process to its corresponding GPU
    torch.cuda.set_device(hvd.local_rank())
  
  model = model_init()
  model = model.to(device)
  
  learning_rate = 5e-5 * hvd.size()
  optimizer  = AdamW(model.parameters(), lr=learning_rate)
  optimizer = hvd.DistributedOptimizer(optimizer, 
                                       named_parameters=model.named_parameters(),
                                       backward_passes_per_step=backward_passes_per_step)
                          
  hvd.broadcast_parameters(model.state_dict(), root_rank=0)
  hvd.broadcast_optimizer_state(optimizer, root_rank=0)
  
  evaluation_statistics = {}
  
  if hvd.rank() == 0:
      print(f"\n{'*' * 5}Training model using Horovod across {total_number_of_gpus} GPUs{'*' * 5}")
      
  for epoch in range(epochs):
    
    with (converter_train.make_torch_dataloader(cur_shard=hvd.rank(), 
                                                shard_count=hvd.size(),
                                                batch_size=train_batch_size) as train_dataloader,
    
          converter_test.make_torch_dataloader(cur_shard=hvd.rank(), 
                                               shard_count=hvd.size(),
                                               batch_size=eval_batch_size) as test_dataloader):
      
      train_dataloader_iter = iter(train_dataloader)
      test_dataloader_iter = iter(test_dataloader)
      
      steps_per_epoch = len(converter_train) // (train_batch_size * hvd.size())
      validation_steps = max(1, len(converter_test) // (eval_batch_size * hvd.size()))
      
      train_one_epoch(model, train_dataloader_iter, optimizer, device, steps_per_epoch, logging_steps, epoch, epochs)
      
      correct_preds_cnt, total_preds_cnt = evaluate(model, test_dataloader_iter, device, steps_per_epoch)
      
      all_preds_cnt = hvd.allgather_object(total_preds_cnt)
      all_correct_cnt = hvd.allgather_object(correct_preds_cnt)
      accuracy = round(sum(all_correct_cnt) / sum(all_preds_cnt), 4)
      
      evaluation_statistics[f"{epoch + 1}"] = accuracy
      
      if hvd.rank() == 0:
        print(f"\Epoch {epoch + 1} evaluation accuracy: {accuracy}")
    
  if hvd.rank() == 0:
    print(f"Evaluation accuracy across all epochs")
    for epoch, accuracy in evaluation_statistics.items():
      print(f"Epoch {epoch} accuracy: {accuracy}")
    
    print(f"Saving trained model {model_save_dir}")
    model.save_pretrained(model_save_dir)

# COMMAND ----------

model_save_dir = 'dbfs:/tmp/petastorm/cache'
dbutils.fs.mkdirs('dbfs:/tmp/petastorm/cache')

args = {'model_save_dir':f"/{model_save_dir.replace(':', '')}", 
        'epochs':3, 
        'train_batch_size':32, 
        'eval_batch_size':32, 
        'backward_passes_per_step':1, 
        'logging_steps':1}

gpu_cnt = 2
hr = HorovodRunner(np=gpu_cnt, driver_log_verbosity='all')
hr.run(main, **args)
