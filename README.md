# Simple SMS Spam Classifier - Production-Ready Skeleton

Building an SMS spam classifier seems straightforward, but presents a series of challenges when applied to real-world
data. The messages are short, and often contain typos, slang, and other non-standard language. Spam messages are
often sent by individuals trying to deceive the recipient, using various tricks to avoid detection.
Also, datasets are often imbalanced, with many more ham messages than spam messages.
While spam messages might be similar, the ham messages might be very different from each
other, leading to having a lot of data with little useful information.

In a production environment, the prediction service must be fast and capable of handling many requests
per second to ensure user satisfaction and maintain usability.
The patterns of spam messages also change over time, necessitating regular model retraining or the use of some form
of continual learning.

This project serves as a skeleton project for a larger enterprise solution that would address the above challenges.
**The goal of this project is to build a simple SMS message spam classifier deployed as a RESTful service in a
Docker container for use as a microservice, serving as a starting point for more complex solutions**.

## Features

We use the [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/)
to train a *Naive Bayes* classifier that predicts the probability that a given SMS message is spam.
The Kaggle website offers a range of well performing and elaborate models, but here we opt for a basic, simple and fast
model.
The dataset contains 5574 SMS messages labeled as either *spam* or *ham*, which are split into a training and testing
set. After basic preprocessing, features are extracted using a vectorizer, TF-IDF, and optionally scaled.
The trained model is then saved and deployed as a RESTful API using Flask and Gunicorn in a Docker container.

**Data Preprocessing**:

- Lowercasing
- Removing punctuation
- Removing stopwords
- Replacing digits with a placeholder
- Removing non-printable characters

**Feature extraction**:

- Bigrams
- Count vectorization
- Filtering out terms by minimum and maximum document frequency
- TF-IDF
- Feature scaling

**API**:

- Single endpoint: `/predict`
- **Input**: JSON with a single key `message` and a string value
- **Output**: JSON with a single key `SPAM_probability` and a float value between 0 and 1

**Deployment**:

- Docker container, emphasis on small image size
- Flask
- Gunicorn

**Confusion Matrix**:

   |                      | Predicted Negative    | Predicted Positive    |
   |----------------------|-----------------------|-----------------------|
   | **Actual Negative**  | 969 (99.18%)          | 8 (0.82%)             |
   | **Actual Positive**  | 9 (6.52%)             | 129 (93.48%)          |


## Installation

1. Ensure you have Python >= 3.6 installed, along with the latest version of Docker.
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data)
   and place the `spam.csv` file in the [dataset](dataset) directory.
3. Set up a virtual environment and install the dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
   Alternatively, use Conda:
    ```bash
    conda create --name smsspamclassifier python=3.11
    conda activate smsspamclassifier
    pip install -r requirements.txt
    ```
   This environment will be used for training the model.

## Usage

1. Set the training parameters in [config/train_config.yaml](config/train_config.yaml).

2. Train the model:
    ```bash
    python src/train_classifier.py <args>
    ```
   To see the available arguments, run:
    ```bash
    python src/train_classifier.py --help
    ```

3. Build the Docker image:
    ```bash
    docker build -t smsspamclassifier -f Dockerfile .
    ```

4. Run the Docker container:
    ```bash
    docker run --rm -it -p 8080:8080 smsspamclassifier
    ```
   The service will now deploy on port 8080 on your local machine.

5. Test the API with the following curl commands:
   ```bash
   curl --request PUT --url http://localhost:8080/predict --header 'content-type: application/json' --data '{"message": "   "}'
   curl --request PUT --url http://localhost:8080/predict --header 'content-type: application/json' --data '{"message": "123456"}'
   curl --request PUT --url http://localhost:8080/predict --header 'content-type: application/json' --data '{"message": "Hey, how have you been? How is Danny doing?"}'
   curl --request PUT --url http://localhost:8080/predict --header 'content-type: application/json' --data '{"message": "Respond to this message to win a free trip to Hawaii"}'
   ```
   Expected outputs are:
   ```bash
   "Empty input - no alphanumeric characters"
   "Malformed input - no alphabetic characters"
   {"SPAM_probability":0.0018944361321917309}
   {"SPAM_probability":0.9146685679649642}
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
