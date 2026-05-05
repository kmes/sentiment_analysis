---
annotations_creators:
- found
language_creators:
- found
language:
- en
license:
- unknown
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets:
- extended|other-tweet-datasets
task_categories:
- text-classification
task_ids:
- multi-class-classification
- sentiment-classification
paperswithcode_id: tweeteval
pretty_name: TweetEval
config_names:
- sentiment
dataset_info:
- config_name: sentiment
  features:
  - name: text
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': negative
          '1': neutral
          '2': positive
  splits:
  - name: train
    num_bytes: 5425122
    num_examples: 45615
  - name: test
    num_bytes: 1279540
    num_examples: 12284
  - name: validation
    num_bytes: 239084
    num_examples: 2000
  download_size: 4849675
  dataset_size: 6943746
configs:
- config_name: sentiment
  data_files:
  - split: train
    path: sentiment/train-*
  - split: test
    path: sentiment/test-*
  - split: validation
    path: sentiment/validation-*
train-eval-index:
- config: sentiment
  task: text-classification
  task_id: multi_class_classification
  splits:
    train_split: train
    eval_split: test
  col_mapping:
    text: text
    label: target
  metrics:
  - type: accuracy
    name: Accuracy
  - type: f1
    name: F1 macro
    args:
      average: macro
  - type: f1
    name: F1 micro
    args:
      average: micro
  - type: f1
    name: F1 weighted
    args:
      average: weighted
  - type: precision
    name: Precision macro
    args:
      average: macro
  - type: precision
    name: Precision micro
    args:
      average: micro
  - type: precision
    name: Precision weighted
    args:
      average: weighted
  - type: recall
    name: Recall macro
    args:
      average: macro
  - type: recall
    name: Recall micro
    args:
      average: micro
  - type: recall
    name: Recall weighted
    args:
      average: weighted
---

# Dataset Card for tweet_eval

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** [Needs More Information]
- **Repository:** [GitHub](https://github.com/cardiffnlp/tweeteval)
- **Paper:** [EMNLP Paper](https://arxiv.org/pdf/2010.12421.pdf)
- **Leaderboard:** [GitHub Leaderboard](https://github.com/cardiffnlp/tweeteval)
- **Point of Contact:** [Needs More Information]

### Dataset Summary

TweetEval consists of seven heterogenous tasks in Twitter, all framed as multi-class tweet classification. This configuration exposes the **sentiment** task only. All datasets have been unified into the same benchmark, with each dataset presented in the same format and with fixed training, validation and test splits.

### Supported Tasks and Leaderboards

- `text_classification`: The dataset can be trained using a SentenceClassification model from HuggingFace transformers.

### Languages

The text in the dataset is in English, as spoken by Twitter users.

## Dataset Structure

### Data Instances

An instance from `sentiment` config:

```
{'label': 2, 'text': '"QT @user In the original draft of the 7th book, Remus Lupin survived the Battle of Hogwarts. #HappyBirthdayRemusLupin"'}
```

### Data Fields

For `sentiment` config:

- `text`: a `string` feature containing the tweet.

- `label`: an `int` classification label with the following mapping:

    `0`: negative

    `1`: neutral

    `2`: positive

### Data Splits

| name      | train | validation | test  |
| --------- | ----- | ---------- | ----- |
| sentiment | 45615 | 2000       | 12284 |

## Dataset Creation

### Curation Rationale

[Needs More Information]

### Source Data

#### Initial Data Collection and Normalization

[Needs More Information]

#### Who are the source language producers?

[Needs More Information]

### Annotations

#### Annotation process

[Needs More Information]

#### Who are the annotators?

[Needs More Information]

### Personal and Sensitive Information

[Needs More Information]

## Considerations for Using the Data

### Social Impact of Dataset

[Needs More Information]

### Discussion of Biases

[Needs More Information]

### Other Known Limitations

[Needs More Information]

## Additional Information

### Dataset Curators

Francesco Barbieri, Jose Camacho-Collados, Luis Espiinosa-Anke and Leonardo Neves through Cardiff NLP.

### Licensing Information

This dataset requires complying with Twitter [Terms Of Service](https://twitter.com/tos) and Twitter API [Terms Of Service](https://developer.twitter.com/en/developer-terms/agreement-and-policy)

Sentiment license: [Creative Commons Attribution 3.0 Unported License](https://groups.google.com/g/semevaltweet/c/k5DDcvVb_Vo/m/zEOdECFyBQAJ)

### Citation Information

```
@inproceedings{barbieri2020tweeteval,
title={{TweetEval:Unified Benchmark and Comparative Evaluation for Tweet Classification}},
author={Barbieri, Francesco and Camacho-Collados, Jose and Espinosa-Anke, Luis and Neves, Leonardo},
booktitle={Proceedings of Findings of EMNLP},
year={2020}
}
```

#### Sentiment Analysis:
```
@inproceedings{rosenthal2017semeval,
  title={SemEval-2017 task 4: Sentiment analysis in Twitter},
  author={Rosenthal, Sara and Farra, Noura and Nakov, Preslav},
  booktitle={Proceedings of the 11th international workshop on semantic evaluation (SemEval-2017)},
  pages={502--518},
  year={2017}
}
```

### Contributions

Thanks to [@gchhablani](https://github.com/gchhablani) and [@abhishekkrthakur](https://github.com/abhishekkrthakur) for adding this dataset.
