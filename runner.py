# Databricks notebook source
# MAGIC %md ### MultiModal runner notebook  
# MAGIC Databricks Runtime 11.3 ML GPU

# COMMAND ----------

pip install pytorch-lightning

# COMMAND ----------

# MAGIC %load_ext autoreload 
# MAGIC %autoreload 2

# COMMAND ----------

import shutil
import horovod.torch as hvd
from sparkdl import HorovodRunner

from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.sql.types import  StructType, StructField, IntegerType, ArrayType
from pyspark.sql.functions import col, pandas_udf
import pyspark.sql.functions as func

import torch
from torch.optim import AdamW
import pytorch_lightning as pl

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

from tokenizer import TableTokenizer
from data_loader import TransformersDataModule
from model import TransformerModel

# COMMAND ----------

# MAGIC %md Tokenize

# COMMAND ----------

MODEL_NAME = "distilbert-base-uncased"

config = {"model_name": MODEL_NAME,
          "batch_size": 1000,
          "truncation": True,
          "padding": "max_length",
          "max_length": 512}

tokenizer = TableTokenizer(**config)

train_df = tokenizer.tokenize_table('imdb_train', 'text')
test_df = tokenizer.tokenize_table('imdb_test', 'text')

# COMMAND ----------

display(train_df)

# COMMAND ----------

bin_size = 25

token_count_distributions = (train_df.selectExpr("size(array_remove(attention_mask, 0)) as token_cnt")
                                     .groupBy('token_cnt').agg(func.count("*").alias('cnt'))
                                     .selectExpr([f"floor(token_cnt / {bin_size}) as token_bin", "token_cnt", "cnt"])
                                     .orderBy(col('token_bin').asc()))

display(token_count_distributions)

# COMMAND ----------

# MAGIC %md Petastorm converter

# COMMAND ----------

# Set a cache directory for Petastorm
petastorm_cache_dir = 'dbfs:/tmp/petastorm/cache'
dbutils.fs.mkdirs(petastorm_cache_dir)

python_format_dir = f"file:///{petastorm_cache_dir.replace(':', '')}"

train_cache_dir = f"{python_format_dir }/train/"
test_cache_dir = f"{python_format_dir }/test/"

# COMMAND ----------

gpu_cnt = 8

spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, train_cache_dir)
converter_train = make_spark_converter(train_df.repartition(gpu_cnt))
               
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, test_cache_dir)
converter_test = make_spark_converter(test_df.repartition(gpu_cnt))

# COMMAND ----------

# MAGIC %md Pytorch Lightning trainer

# COMMAND ----------

def train(model, dataloader, epochs, steps_per_epoch, gpus=0, strategy=None, device_id=0, device_count=1, ckpt_restore=None, 
          default_dir="/dbfs/tmp/petastorm/model"):
  
  trainer = pl.Trainer(gpus=gpus, 
                       max_epochs=epochs, 
                       limit_train_batches=steps_per_epoch, 
                       log_every_n_steps=1,
                       val_check_interval=steps_per_epoch,
                       num_sanity_val_steps=0,
                       limit_val_batches=1,
                       reload_dataloaders_every_n_epochs=1,
                       strategy=strategy,
                       default_root_dir = default_dir)
  
  trainer.fit(model, dataloader, ckpt_path=ckpt_restore)
  
  return model.model if device_id == 0 else None

# COMMAND ----------

# MAGIC %md Horovod trainer

# COMMAND ----------

def train_hvd(model_name, model_config, converter_train, converter_test, batch_size, epochs):
  hvd.init()

  if hvd.rank() == 0:
    print(f"Horovod rank: {hvd.rank()} and size: {hvd.size()}")
    
  BATCH_SIZE = 32
  STEPS_PER_EPOCH = len(converter_train)*hvd.size() //  BATCH_SIZE
    
  hvd_model = TransformerModel(MODEL_NAME, 
                               model_config, 
                               lr=1e-5*hvd.size(), 
                               device_id=hvd.rank(), 
                               device_count=hvd.size())
  
  hvd_datamodule = TransformersDataModule(converter_train, 
                                          converter_test,
                                          batch_size=BATCH_SIZE,
                                          device_id=hvd.rank(),
                                          device_count=hvd.size())
  
  return train(hvd_model, hvd_datamodule, epochs, STEPS_PER_EPOCH, gpus=1, device_id=hvd.rank(), device_count=hvd.size())

# COMMAND ----------

# MAGIC %md Train

# COMMAND ----------

labels_table_name = 'imdb_labels'
labels = spark.table(labels_table_name)
collected_labels = labels.collect()
 
id2label = {row.idx: row.label for row in collected_labels} 
label2id = {row.label: row.idx for row in collected_labels}

model_config = AutoConfig.from_pretrained(MODEL_NAME, 
                                          num_labels=2,
                                          id2label=id2label, 
                                          label2id=label2id,
                                          problem_type="single_label_classification")

args = {'model_name':MODEL_NAME, 
        'model_config':model_config, 
        'converter_train':converter_train,
        'converter_test':converter_test,
        'batch_size': 32,
        'epochs': 5}

pyfile_dependencies = ['model.py', 'data_loader.py']

for file in pyfile_dependencies:
  driver_node_copy = f'/{file}'
  shutil.copyfile(file, driver_node_copy)
  spark.sparkContext.addPyFile(driver_node_copy)

# COMMAND ----------

hr = HorovodRunner(np=2, driver_log_verbosity='all')
hvd_model = hr.run(train_hvd, **args)
