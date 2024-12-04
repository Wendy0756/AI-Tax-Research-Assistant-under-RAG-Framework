# AI-Tax-Research-Assistant-under-RAG-Framework
This study aims to:
* Compare and identify the optimal embedding model for the tax domain between Legal-BERT, an embedding model pre-trained with legal documents, and "all-MiniLM-L6-v2", an efficient and powerful sentence embedding model.
* Unveil the effectiveness of domain-specific fine-tuning in RAG framework
* Analyze the relationship between model performance metrics - semantic similarity (demonstrated by BERTSCores), relevance and faithfulness (measures by RAGAS score)- and key parameters, including corpus chunk size, token overlap, and LLM temperature.

## Environment: 
* Google Colab, A100 Nvidia GPU

## Installation:
* !pip3 config set global.quiet false --quiet
* !pip3 install datasets -quiet
* !pip3 install trl --quiet
* !pip3 install -U langchain-huggingface --quiet
* !pip3 install langchain_community --quiet
* !pip3 install pymupdf --quiet
* !pip3 install qdrant-client --quiet
* !pip3 install beautifulsoup4 --quiet
* !pip3 install -q -U bitsandbytes --quiet
* !pip3 install -q -U transformers accelerate --quiet
* !pip3 install bert-score transformers --quiet
* !pip3 install ragas --quiet
* !pip install --upgrade openai --quiet

## Packages:
For data loading, pre-processing and visualization:
* import pandas as pd
* import os
* import re
* from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
* from google.colab import userdata
* from google.colab import drive
* drive.mount('/content/drive')
* nltk.download('punkt')
* nltk.download('stopwords')
* import seaborn as sns
* import matplotlib.pyplot as plt

For LLM fine-tuning:
* import bitsandbytes
* from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
* SFTTrainerfrom trl import SFTTrainerSFTTrainerfrom trl import SFTTrainer
* import torch
* import transformers
* from transformers import(
  AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    pipeline,
    BitsAndBytesConfig,
    Trainer,
    AutoModelForPreTraining
)

For RAG framework:
* from bs4 import SoupStrainer
* from langchain_text_splitters import RecursiveCharacterTextSplitter
* from langchain_community.vectorstores import Qdrant
* from langchain_community.document_loaders import WebBaseLoader
* from langchain_community.document_loaders import TextLoader
* from langchain_community.document_loaders import OnlinePDFLoader
* from langchain_community.document_loaders import PyMuPDFLoader
* from langchain_community.document_loaders import PubMedLoader
* from langchain_community.embeddings import HuggingFaceEmbeddings
* from langchain_core.prompts import ChatPromptTemplate
* from langchain_core.runnables import RunnablePassthrough
* from langchain_core.output_parsers import StrOutputParser
* from langchain_huggingface import HuggingFacePipeline
* from langchain.llms import HuggingFacePipeline

For evaluation:
* import openai
* from ragas import evaluate, metrics
* from bert_score import score as bert_score

## Datasets:
* https://huggingface.co/datasets/AbrorKh/tax_data_sample
* https://huggingface.co/datasets/mdacampora/tax-chat-as-dicts/viewer
