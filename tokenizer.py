import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import  StructType, StructField, IntegerType, ArrayType
from typing import Optional, Union, List
from transformers import AutoTokenizer

spark = SparkSession.builder.getOrCreate()


class TableTokenizer():
  def __init__(self, model_name:str, batch_size=1000, truncation=True, padding:Union[bool, str]='max_length', max_length=512):
    self.model_name = model_name
    self.batch_size = batch_size
    self.truncation = truncation
    self.padding = padding
    self.max_length = max_length
    self.tokenizer = self._init_tokenizer(truncation, padding, max_length)
    

  def _init_tokenizer(self, truncation, padding, max_length):
    
    tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
    
    def tokenize(batch):
      return tokenizer(batch, 
                      truncation=truncation,
                      padding=padding,
                      max_length=max_length)
    
    return tokenize
    
    
  def get_tokenizer_udf(self):

    schema = StructType()
    schema.add(StructField('input_ids', ArrayType(IntegerType()), True))
    schema.add(StructField('attention_mask', ArrayType(IntegerType()), True))
    
    tokenize = self.tokenizer

    def tokenize_udf(df:pd.DataFrame) -> pd.DataFrame:

      tokenized = df.apply(lambda x: tokenize(x))

      tokenized = tokenized.to_frame(name='tokenized')

      tokenized['input_ids'] = tokenized.tokenized.apply(lambda x: x['input_ids'])

      tokenized['attention_mask'] = tokenized.tokenized.apply(lambda x: x['attention_mask'])

      return tokenized[['input_ids', 'attention_mask']]

    return pandas_udf(tokenize_udf, returnType=schema)
  
  
  def tokenize_table(self, delta_table_name:str, text_col:str, n_samples:Union[bool, str]=None):
  
    tokenizer = self.get_tokenizer_udf()

    text_df = spark.table(delta_table_name).limit(n_samples) if n_samples else spark.table(delta_table_name)

    tokenized_df = (text_df.withColumn('tokenized', tokenizer(col(text_col)))
                           .selectExpr("tokenized.input_ids as input_ids", 
                                       "tokenized.attention_mask as attention_mask",
                                       "label as labels"))

    return tokenized_df
  



  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  