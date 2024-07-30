import argparse
import pickle
import re
import time
from argparse import Namespace
from typing import Tuple, List, Any

import numpy
import pandas
from loguru import logger
from omegaconf import OmegaConf, DictConfig
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

import const as util


def load_kaggle_sms_spam_dataset(dataset_path: str) -> pandas.DataFrame:
    """
    Load the Kaggle SMS spam dataset into a pandas DataFrame.

    The DataFrame will contain the following columns:
    - "label" (str): The label of the message, either "spam" or "ham".
    - "message" (str): The content of the SMS message.
    - "label_num" (int): A numerical representation of the label, where 0 is "ham" and 1 is "spam".

    Args:
        dataset_path (str): Path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: DataFrame containing the loaded dataset.
    """
    sms_messages = pandas.read_csv(dataset_path, encoding=util.KAGGLE_SMS_FILE_ENCODING)
    sms_messages.dropna(how="any", inplace=True, axis=1)
    sms_messages.columns = ["label", "message"]
    sms_messages["label_num"] = sms_messages.label.map({"ham": 0, "spam": 1})

    return sms_messages


def replace_digits_in_message(message: str, placeholder: str) -> str:
    """
    Replaces all digits in the given message with the specified placeholder.

    Args:
        message (str): The message to process.
        placeholder (str): The string to replace the digits with.

    Returns:
        str: The message with digits replaced by the placeholder.
    """
    return re.sub(r"\d+", placeholder, str(message))


def clean_data_frame_spam(
        messages: List[str], label_nums: List[int]
) -> pandas.DataFrame:
    """
    Cleans the given messages and labels by replacing digits in the messages vectorizer
    and removing non-printable characters. Only English messages are kept, 
    and the filtered messages and labels are returned as a pandas DataFrame.

    Args:
        messages (List[str]): A list of messages to be cleaned.
        label_nums (List[int]): A list of corresponding label numbers.

    Returns:
        pandas.DataFrame: A DataFrame containing the cleaned messages and labels.

    Raises:
        None
    """
    parsed_messages = [
        replace_digits_in_message(message=msg.strip(), placeholder=util.NUMBER_PLACEHOLDER)
        for msg in messages
    ]

    filtered_messages, filtered_labels = list(), list()
    for pm, lab in zip(parsed_messages, label_nums):
        printable_str = "".join(x for x in pm if x.isprintable())

        filtered_messages.append(printable_str)
        filtered_labels.append(lab)

    return pandas.DataFrame(
        data={"message": filtered_messages, "label_num": filtered_labels}
    )


def get_vectorizer(max_df: float, min_df: int, bigrams: bool) -> CountVectorizer:
    """
    Create and return a CountVectorizer object with specified parameters.

    Args:
        max_df (float): The maximum document frequency threshold. Ignore 
            terms that appear in more than max_df fraction of the documents.
        min_df (int): The minimum document frequency threshold. Only 
            keep terms that appear in at least min_df documents.
        bigrams (bool): A flag indicating whether to include bigrams in 
            the vectorizer.

    Returns:
        CountVectorizer: A CountVectorizer object with the specified parameters.
    """
    if bigrams:
        ngram_range: Tuple[int, int] = (1, 2)
    else:
        ngram_range: Tuple[int, int] = (1, 1)

    return CountVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
    )


def create_pipeline(vectorizer: CountVectorizer, tf_idf: bool, scale: bool) -> Pipeline:
    """
    Create a pipeline for training a spam classifier.

    Parameters:
    - vectorizer (CountVectorizer): The vectorizer to use for text feature extraction.
    - tf_idf (bool): Whether to apply TF-IDF transformation to the features.
    - scale (bool): Whether to scale the features using MaxAbsScaler.

    Returns:
    - Pipeline: The pipeline object for training the spam classifier.
    """
    steps: List[Tuple[str, Any]] = [("vectorizer", vectorizer)]

    if tf_idf:
        steps.append(("tfidf", TfidfTransformer()))

    if scale:
        steps.append(("scaler", MaxAbsScaler()))

    steps.append(("classifier", MultinomialNB()))

    return Pipeline(steps=steps)


def model_test_supervised(
        pipeline: Pipeline, x_test: pandas.Series, y_test: pandas.Series
) -> None:
    """
    Test the supervised model using the provided pipeline, test data, and labels.

    Args:
        pipeline (Pipeline): The trained pipeline for making predictions.
        x_test (pandas.Series): The test data.
        y_test (pandas.Series): The true labels for the test data.

    Returns:
        None

    """
    logger.info("Testing model")

    # Calculate predictions and record the time required
    start_time: float = time.time()
    test_predictions: numpy.ndarray = pipeline.predict(x_test)
    end_time: float = time.time()

    # Evaluate metrics
    accuracy_score: float = metrics.accuracy_score(y_test, test_predictions)
    confusion_matrix: numpy.ndarray = metrics.confusion_matrix(y_test, test_predictions)
    confusion_matrix_normalized: numpy.ndarray = metrics.confusion_matrix(
        y_test, test_predictions, normalize="true"
    )

    # Calculate predicted probabilities for X_test_dtm (poorly calibrated)
    # and calculate AUC
    y_pred_prob: numpy.ndarray = pipeline.predict_proba(x_test)[:, 1]
    area_under_curve: float = metrics.roc_auc_score(y_test, y_pred_prob)

    # Report metrics to user
    logger.info(f"Accuracy: {accuracy_score}")
    logger.info(f"Confusion matrix: \n{confusion_matrix}")
    logger.info(f"Normalized confusion matrix: \n{confusion_matrix_normalized * 100}")
    logger.info(f"AUC: {area_under_curve}")

    # Get time difference in seconds and report time stats to user
    time_diff: float = end_time - start_time
    logger.info(
        f"{x_test.shape[0]} messages took {time_diff} seconds, "
        f"which is {x_test.shape[0] / time_diff:.2f} msg/s"
    )


def train(
        x_train: pandas.Series,
        y_train: pandas.Series,
        max_df: float,
        min_df: int,
        bigrams: bool = True,
        tfidf: bool = False,
        scale: bool = False,
) -> Pipeline:
    """
    Trains a spam classifier pipeline.

    Args:
        x_train (pandas.Series): The input training data.
        y_train (pandas.Series): The target training data.
        max_df (float): The maximum document frequency for the CountVectorizer.
        min_df (int): The minimum document frequency for the CountVectorizer.
        bigrams (bool, optional): Whether to include bigrams in the vectorization process. Defaults to True.
        tfidf (bool, optional): Whether to apply TF-IDF transformation to the vectorized data. Defaults to False.
        scale (bool, optional): Whether to scale the data. Defaults to False.

    Returns:
        Pipeline: The trained spam classifier pipeline.
    """
    logger.info("Setting up pipeline")

    vectorizer: CountVectorizer = get_vectorizer(
        max_df=max_df, min_df=min_df, bigrams=bigrams
    )
    pipeline: Pipeline = create_pipeline(
        vectorizer=vectorizer, tf_idf=tfidf, scale=scale
    )

    logger.info("Fitting data to pipeline")
    pipeline.fit(x_train, y_train)
    logger.info("Done fitting")

    return pipeline


def main(
        dataset_path: str,
        output_path: str,
        training_config: DictConfig
) -> None:
    """
    Main function to load data, clean it, train a spam classifier, and save the trained model.

    Args:
        dataset_path (str): Path to the dataset for training/testing.
        output_path (str): Path to save the trained model.
        training_config (dict): Configuration dictionary for training parameters.

    Returns:
        None
    """
    logger.info("Loading data")
    dataset: pandas.DataFrame = load_kaggle_sms_spam_dataset(
        dataset_path=dataset_path
    )

    logger.info("Cleaning data")
    dataset = clean_data_frame_spam(
        messages=dataset["message"].tolist(),
        label_nums=dataset["label_num"].tolist()
    )

    messages: pandas.Series = dataset.message
    label_numbers: pandas.Series = dataset.label_num

    logger.info(
        f"Splitting data into train and test with percentages "
        f"{1 - training_config['test_data_percentage']}/{training_config['test_data_percentage']}"
    )
    x_train, x_test, y_train, y_test = train_test_split(
        messages,
        label_numbers,
        train_size=1 - training_config["test_data_percentage"],
        random_state=training_config["seed"],
    )

    logger.info(f"Number of messages: {messages.shape[0]}")
    logger.info(f"Train/test set sizes: {x_train.shape[0]}/{x_test.shape[0]}")
    pipeline: Pipeline = train(
        x_train=x_train,
        y_train=y_train,
        max_df=training_config["max_df"],
        min_df=training_config["min_df"],
        bigrams=training_config["bigrams"],
        tfidf=training_config["tfidf"],
        scale=training_config["scale"],
    )
    model_test_supervised(pipeline=pipeline, x_test=x_test, y_test=y_test)

    # logger.info("Training model on whole dataset")
    # pipeline: Pipeline = train(
    #     x_train=messages,
    #     y_train=label_numbers,
    #     max_df=training_config["max_df"],
    #     min_df=training_config["min_df"],
    #     bigrams=training_config["bigrams"],
    #     tfidf=training_config["tfidf"],
    #     scale=training_config["scale"],
    # )

    logger.info(f"Saving pipeline to {output_path}")
    with open(output_path, "wb") as output_stream:
        pickle.dump(pipeline, output_stream)

    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-path",
        type=str,
        default="dataset/spam.csv",
        help="path to the dataset for training/testing",
    )
    parser.add_argument("--output-path", type=str, default="model/model.pkl")
    parser.add_argument("--train-config", type=str, default="config/train_config.yaml")
    args: Namespace = parser.parse_args()

    train_config = OmegaConf.load(args.train_config)

    main(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        training_config=train_config
    )
