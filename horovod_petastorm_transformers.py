# Databricks notebook source
# MAGIC %md ### Scaling transformer model training across a GPU cluster using Horovod and Petastorm  
# MAGIC This notebook provides an example implementation of [Horovod](https://horovod.readthedocs.io/en/stable/) for distributed training of transformer models. It was developed using Databricks ML Runtime 11.2 GPU. Training and testing datasets are derived from Delta tables cached using [Petastorm](https://petastorm.readthedocs.io/en/latest/index.html). Petastorm allows for easy conversion between parquet files and Pytorch Dataloaders that iterate through parquet row groups and allow for efficient columnar access. Horovod provides flexible scaling, allowing users to execute training loops on a single server with one or more GPUs or a cluster of servers, each with one or more GPUs. For lower cost, consider scaling across a cluster of single-GPU-backed instances.
# MAGIC 
# MAGIC Data pre-processing is handled by Spark. This includes applying the transformer model's tokenizer directly to a Spark DataFrame via a PanadasUDF. The tokenized DataFrame is then cached using Petastorm.
# MAGIC 
# MAGIC Horovod has two methods for executing training loops, the higher-level Estimator API and the more fine-grained, standard Pytorch training loop method. This example is based on the latter approach. For examples of both approaches provided by Horovod, see the links below.
# MAGIC 
# MAGIC **Getting started**  
# MAGIC To create the IMDB Delta tables referenced in this notebook, follow steps 1 and 2 of the "Getting started" section of [this repository](https://github.com/marshackVB/rapid_nlp_blog). Steps 2 will download additional datasets for text classification problems. You can optionally run cells only associated with the IMDB data ingestion and table creation.  
# MAGIC 
# MAGIC **Other useful resources**  
# MAGIC  - Example [pytorch training loop implementation](https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_mnist.py#L71) from Horovod  
# MAGIC  - Example [TorchEstimator implementation](https://github.com/horovod/horovod/blob/master/examples/spark/pytorch/pytorch_spark_mnist.py) from Horovod
# MAGIC  - Horovod [Spark documentation](https://horovod.readthedocs.io/en/stable/spark_include.html) and [Pytorch documentation](https://horovod.readthedocs.io/en/stable/pytorch.html)
# MAGIC  - DAIS [talk on petastorm](https://www.youtube.com/watch?v=lQJO_aKGaFs) with [example petastorm/horovod implementation](https://docs.databricks.com/_static/notebooks/deep-learning/petastorm-spark-converter-pytorch.html)

# COMMAND ----------

import horovod.torch as hvd
import horovod.spark
import horovod
from sparkdl import HorovodRunner

import pandas as pd
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.sql.types import  StructType, StructField, IntegerType, ArrayType
from pyspark.sql.functions import col, pandas_udf

import torch
from torch.optim import AdamW

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

# COMMAND ----------

# MAGIC %md #### Create a PandasUDF that apply a huggingface tokenizer to a Spark DataFrame column

# COMMAND ----------

model_type = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=True)

#Broadcast the tokenzer to the cluster's worker nodes
spark.sparkContext.broadcast(tokenizer)

# COMMAND ----------

# MAGIC %md Define a PandasdUDF to apply the tokenizer

# COMMAND ----------

def tokenize_udf(df:pd.DataFrame) -> pd.DataFrame:
  """A PandasUDF to tokenize a text column of a Spark Dataframe
  """

  def tokenize(batch):
    """Tokenize input text, returning input_ids and attention_mask
    lists. Set the max_length number as low as possible without
    removing a significant amount of input text tokens. The limit is
    512 for distilbert-base-uncased.
    """

    return tokenizer(batch, 
                     truncation=True,
                     padding='max_length',
                     max_length=300)

  tokenized = df.apply(lambda x: tokenize(x))

  tokenized = tokenized.to_frame(name='tokenized')

  tokenized['input_ids'] = tokenized.tokenized.apply(lambda x: x['input_ids'])

  tokenized['attention_mask'] = tokenized.tokenized.apply(lambda x: x['attention_mask'])

  return tokenized[['input_ids', 'attention_mask']]

# COMMAND ----------

# MAGIC %md Apply the UDF to the text column of the Spark DataFrame.

# COMMAND ----------

# Define the Spark Schema that maps to the Pandas DataFrame schema returned by the PandasUDF
schema = StructType()
schema.add(StructField('input_ids', ArrayType(IntegerType()), True))
schema.add(StructField('attention_mask', ArrayType(IntegerType()), True))

tokenize_pandas_udf = pandas_udf(tokenize_udf, returnType=schema)

train_df = (spark.table('imdb_train').withColumn('tokenized', tokenize_pandas_udf(col('text')))
                 .selectExpr('tokenized.input_ids as input_ids', 'tokenized.attention_mask as attention_mask', 'label as labels'))
                 #.limit(5000))

test_df = (spark.table('imdb_test').withColumn('tokenized', tokenize_pandas_udf(col('text')))
               .selectExpr('tokenized.input_ids as input_ids', 'tokenized.attention_mask as attention_mask', 'label as labels'))
               #.limit(1000))

# COMMAND ----------

display(train_df)

# COMMAND ----------

# MAGIC %md #### Convert the Spark DataFrames to a Petastorm dataset  
# MAGIC See [petastorm make_spark_converter documentation](https://petastorm.readthedocs.io/en/latest/api.html#petastorm.spark.spark_dataset_converter.make_spark_converter)

# COMMAND ----------

# Set a cache directory for Petastorm
petastorm_cache_dir = 'dbfs:/tmp/petastorm/cache'
dbutils.fs.mkdirs(petastorm_cache_dir)

python_format_dir = f"file:///{petastorm_cache_dir.replace(':', '')}"

train_cache_dir = f"{python_format_dir }/train/"
test_cache_dir = f"{python_format_dir }/test/"

# COMMAND ----------

# Repartition the DataFrame to a number >= # of GPUs
gpu_cnt = 8

# Cache the tables
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, train_cache_dir)
converter_train = make_spark_converter(train_df.repartition(gpu_cnt))
               
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, test_cache_dir)
converter_test = make_spark_converter(test_df.repartition(gpu_cnt))

# COMMAND ----------

# View the parquet files generated by Petastorm
converter_train.file_urls

# COMMAND ----------

# View the count of rows
converter_train.dataset_size

# COMMAND ----------

# MAGIC %md Delete the petastorm datasets

# COMMAND ----------

# This is useful to execute after model training completes
# converter_train.delete()
# converter_test.delete()

# COMMAND ----------

# MAGIC %md Delete all datasets in the cached directory using dbutils

# COMMAND ----------

# This is useful if cached datasets accrue over time and you want to clear them out
#dbutils.fs.rm(train_cache_dir, recurse=True)
#dbutils.fs.rm(test_cache_dir, recurse=True)

# COMMAND ----------

# MAGIC %md #### Create a transformer model initializer to return a new model instance

# COMMAND ----------

labels_table_name = 'imdb_labels'
labels = spark.table(labels_table_name)
collected_labels = labels.collect()
 
id2label = {row.idx: row.label for row in collected_labels} 
label2id = {row.label: row.idx for row in collected_labels}

def model_init(model_type_or_path=model_type, id2label=id2label, label2id=label2id, 
               problem_type="single_label_classification", num_labels=2):
  
  model_config = AutoConfig.from_pretrained(model_type_or_path, 
                                            num_labels=num_labels,
                                            id2label=id2label, 
                                            label2id=label2id,
                                            problem_type=problem_type)
  
  return AutoModelForSequenceClassification.from_pretrained(model_type_or_path, 
                                                            config=model_config)

# COMMAND ----------

# MAGIC %md #### Specify Pytorch training and evaluation loops 

# COMMAND ----------

def train_one_epoch_mixed_precision(model, dataloader, optimizer, device, scaler, steps_per_epoch, logging_steps):
  """A single epoch training loop with mixed precision training.
  
  See example from horovod: https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_mnist.py#L71
  and pytorch documentation: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#all-together-automatic-mixed-precision
  """
  model.train
  for step in range(steps_per_epoch):
    if hvd.rank() == 1 and step % logging_steps == 0: 
      print(f"Running step {step} out of {steps_per_epoch}")
    batch = next(dataloader)
    batch = {k: v.to(device) for k, v in batch.items()}
    
    optimizer.zero_grad()
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
      output = model(**batch)
      loss = output.loss
      
    scaler.scale(loss).backward()
    optimizer.synchronize()
    scaler.unscale_(optimizer)
    
    with optimizer.skip_synchronize():
      scaler.step(optimizer)
          
    scaler.update()

    
def train_one_epoch(model, dataloader, optimizer, device, steps_per_epoch, logging_steps):
  """A single epoch training loop
  """
  model.train
  for step in range(steps_per_epoch):
    if hvd.rank() == 1 and step % logging_steps == 0: 
        print(f"Running step {step} out of {steps_per_epoch}")
    batch = next(dataloader)
    batch = {k: v.to(device) for k, v in batch.items()}
    
    optimizer.zero_grad()
    
    output = model(**batch)
    loss = output.loss
    loss.backward()
    optimizer.step()
    
    
def evaluate(model, dataloader, device, steps_per_epoch):
  
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

# MAGIC %md #### Create a function that incorporates Horovod into the training and evaluation loops  

# COMMAND ----------

def main(model_save_dir, epochs=3, train_batch_size=32, eval_batch_size=64, backward_passes_per_step=4, logging_steps=10, use_amp=True):
  """This function will run on each horovod process, with one process running per GPU.
  """
  
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
      
  if use_amp:
    scaler = torch.cuda.amp.GradScaler()
  
  for epoch in range(epochs):
      
    if hvd.rank() == 0:
      print(f"\nTraining epoch {epoch + 1} / {epochs}")
    
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
      
      if use_amp:         
        train_one_epoch_mixed_precision(model, train_dataloader_iter, optimizer, device, scaler, steps_per_epoch, logging_steps)
      else:
        train_one_epoch(model, train_dataloader_iter, optimizer, device, steps_per_epoch, logging_steps)
      
      if hvd.rank() == 0:
        print(f"\nRunning evaluation")
      
      correct_preds_cnt, total_preds_cnt = evaluate(model, test_dataloader_iter, device, steps_per_epoch)
      
      all_preds_cnt = hvd.allgather_object(total_preds_cnt)
      all_correct_cnt = hvd.allgather_object(correct_preds_cnt)
      accuracy = round(sum(all_correct_cnt) / sum(all_preds_cnt), 4)
      
      evaluation_statistics[f"epoch_{epoch}"] = accuracy
      
      if hvd.rank() == 0:
        print(f"\nAccuracy: {accuracy}")
    
  if hvd.rank() == 0:
    
    for epoch, accuracy in evaluation_statistics.items():
      print(f"\nAccuracy across all epochs")
      print(f"{epoch} accuracy: {accuracy}")
    
    print(f"Saving trained model {model_save_dir}")
    model.save_pretrained(model_save_dir)

# COMMAND ----------

# MAGIC %md #### Training configuration

# COMMAND ----------

model_save_dir = 'dbfs:/tmp/petastorm/cache'
dbutils.fs.mkdirs('dbfs:/tmp/petastorm/cache')

args = {'model_save_dir':f"/{model_save_dir.replace(':', '')}", 
        'epochs':1, 
        'train_batch_size':32, 
        'eval_batch_size':64, 
        'backward_passes_per_step':4, 
        'logging_steps':10, 
        'use_amp':True}

# COMMAND ----------

# MAGIC %md #### Single node training

# COMMAND ----------

hr = HorovodRunner(np=-1, driver_log_verbosity='all')
hr.run(main, **args)

# COMMAND ----------

# MAGIC %md #### Mutli-node training

# COMMAND ----------

gpu_cnt = 2
hr = HorovodRunner(np=gpu_cnt, driver_log_verbosity='all')
hr.run(main, **args)

# COMMAND ----------

# MAGIC %md #### View model artifacts directory

# COMMAND ----------

dbutils.fs.ls(model_save_dir)

# COMMAND ----------

# MAGIC %md #### Load the trained model

# COMMAND ----------

model = model_init(args['model_save_dir'])
