from pathlib import Path
import shutil
import os
import logging
import sys

from transformers import AutoModelForSequenceClassification

from finbert.finbert import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from detoxify import Detoxify
import argparse

def main(args):

    pd.set_option('max_colwidth', -1)

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.ERROR)

    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    topic_classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-multi-all", return_all_scores=True)

    def get_emotion_scores(texts: list):
        return [{ x['label'] : x['score'] for x in pred } for pred in emotion_classifier(texts)]

    def get_topic_scores(texts: list):
        return [{ x['label'].replace('&', 'and') : x['score'] for x in pred } for pred in topic_classifier(texts)]

    def extract_features(finbert_result: pd.DataFrame) -> pd.DataFrame:
        emotion_scores_df = pd.json_normalize(get_emotion_scores(list(finbert_result.tokens)))
        finbert_result = finbert_result.join(emotion_scores_df)
        topic_scores_df = pd.json_normalize(get_topic_scores(list(finbert_result.tokens)))
        finbert_result = finbert_result.join(topic_scores_df)
        
        return finbert_result
    lm_path = os.path.join(args.model_path, 'language_model/finbertTRC2')
    cl_path = os.path.join(args.model_path, 'classifier_model/finbert-sentiment')
    cl_data_path = args.data_path
    bertmodel = AutoModelForSequenceClassification.from_pretrained(lm_path,cache_dir=None, num_labels=3)


    config = Config(data_dir=cl_data_path,
                    bert_model=bertmodel,
                    num_train_epochs=4,
                    model_dir=cl_path,
                    max_seq_length = 48,
                    train_batch_size = 32,
                    learning_rate = 2e-5,
                    output_mode='classification',
                    warm_up_proportion=0.2,
                    local_rank=-1,
                    discriminate=True,
                    gradual_unfreeze=True)
    
    
    finbert = FinBert(config)
    finbert.base_model = 'bert-base-uncased'
    finbert.config.discriminate=True
    finbert.config.gradual_unfreeze=True
    finbert.prepare_model(label_list=['positive','negative','neutral'])
    test_data = finbert.get_data('test')
    train_data = finbert.get_data('train')
    val_data = finbert.get_data('validation')
    model = AutoModelForSequenceClassification.from_pretrained(cl_path, cache_dir=None, num_labels=3)
    model.to(finbert.device)

    train_res = finbert.evaluate(model=model, examples=train_data)
    train_res['prediction'] = train_res.predictions.apply(lambda x: np.argmax(x,axis=0))
    train_res = train_res.drop(columns=['agree_levels', 'predictions'])

    val_res = finbert.evaluate(model=model, examples=val_data)
    val_res['prediction'] = val_res.predictions.apply(lambda x: np.argmax(x,axis=0))
    val_res = val_res.drop(columns=['agree_levels', 'predictions'])

    test_res = finbert.evaluate(model=model, examples=test_data)
    test_res['prediction'] = test_res.predictions.apply(lambda x: np.argmax(x,axis=0))
    test_res = test_res.drop(columns=['agree_levels', 'predictions'])


    features_train = extract_features(train_res)
    features_train.to_csv(os.path.join(cl_data_path,'finbert_train_features.csv'), sep='\t')


    features_val = extract_features(val_res)
    features_val.to_csv(os.path.join(cl_data_path, 'finbert_val_features.csv'), sep='\t')


    features_test = extract_features(test_res)
    features_test.to_csv(os.path.join(cl_data_path, 'finbert_test_features.csv'), sep='\t')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='pre_process_and_train_data.py [<args>] [-h | --help]'
    )
    
    
    parser.add_argument('--data_path', type=str, default=None, help='used for resume')
    parser.add_argument('--model_path', type=str, default=None, help='used for resume')
    

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    random_seed = 0
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args = parse_args()
    main(args)