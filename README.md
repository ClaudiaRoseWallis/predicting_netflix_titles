# Predicting Netflix Titles
This repo contains code to predict which Netflix titles a user may want to watch.

## Context
**Project Name:** Predict Netflix titles that Mum will watch       
**Written by:** Claudia Wallis         
**Created:** 27/02/2022           
**Last modifed:** 27/02/2022           

## Objective
**Predict which Netflix Titles Mum will want to watch in England**   

## Input Data
1. [netflix_dataset_latest_2021_kaggle.xlsx](https://www.kaggle.com/syedmubarak/netflix-dataset-latest-2021)
2. Mum's Netflix History: downloaded from Netflix

## Outcome 
A XGBoost Model was built after comparing multiple algorithms and tuning parameters. Unfortunately it overfitted and so its performance on the test set is not as strong as that on the train & validation. However it is better than random and the feature importance makes sense given my understanding of what Mum likes to watch. Unfortunately the dataset was extremely unbalanced and so we had to oversample using SMOTE. More Netflix preference data and exploration of other upsampling methods could see improved results.
