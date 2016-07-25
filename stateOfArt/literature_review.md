# Literature review

## Contents
- 1. Introduction
- 1.1 Problem statement
- 1.2 Motivations
- 3. Literature Review
- 6. References


## WIP notes

TODOs:
- judgement at first sight (how personality fit plays a role in professional envs);

Questions to answer:
- personality inference
- can personality assessment be useful as a feature space for inference/prediction?
- Video feature space versus Personality assessment?

Note: dataset "The YouTube personality dataset" - https://www.idiap.ch/dataset/youtube-personality


## Introduction 


The present work focuses on computational personality recognition systems 


## Problem statement 

Traditional approaches of personality assessment require individuals to answer predefined questionaires/surveys, which can be both time-consuming and, most of all, impratical in the majority of business contexts [2].

As R. Hu and P. Pu [P9] state, distinct attempts - both implicit and explicit - have been conducted in industry context to collect personality information in order to use it for recommenadation: "For example, Whattorent.com, a movie recommender system, recommends
movies based on users’ personality measured by 20 scene-oriented personality questions. The detailed introduction can be found in [11]. Yobo.com is a Chinese music recommender website, providing personality quizzes to infer users’ “music DNA” or users’ musical preferences. In addition, some online commerce websites, such as Gifts.com, are emerging to make gift suggestions based on recipients’ personality measured by personality quizzes, with the aim of facilitating the gift selection process."    


As stated in [8], analysis of video content appears to be the least studied problems in the domain of computational personality recognition.

On the other hand, one of the most common application of machine learning techniques in business context are recommendation systems. One of the most common problems in building an efficient recommendation system is the cold-start problem, where the non-existance of considerable amount of data revents the recommendation system to perform well. 
The application of personality based features as initial inputs could help addressing and potentially overcoming this issue [P9].

## Motivation

Extensive research suggests that knowing personality traits of individuals can be useful as a predictor of preferences on decision making for websites, content, products, brands, and services [P2].
Recently studies have shown that automatic personality trait recognition of individuals based on user generated content on social media platforms - such as Facebook, Youtube, Tweeter, etc. - can be useful predictors [P8]. For example, research has shown that automated personality inference based on Facebook likes to be more accurate than those made by users' friends and spouses [P2]. Besides textual based analysis, other studies have shown relationship on level of extraversion of an individual and the number of friends he has and number of mobile phone calls he makes.  
Furthermore, research has also been developed to improve efficiency of recommendation systems based on personality traits. Studies have shown that collaborative filtering systems - one of the most successful and widely implemented recommendation algorithms - could be enhanced by combining personality information with user rating information [P9].
R. Hu and P. Pu [P9] highlight as well the benefits of improved accuracy specially in the case of cold-start problem.  


## State-of-the-art
TODO

### Emotion detection
TODO

#### based on image
TODO

#### based on video

facial expression analysis toolkit (FACET)

Facial Action Coding System (FACS)

Quote from [P10]:
----
"In relevant studies [15], researchers employed webcams to record the responses of job applicants to work-related prompts in the context of social work behavior. A high correlation was found between the webcam test scores and job placement success. Multimodal cues have also been found to affect interviewers’ judgments, and they have been found to be a positive indicator of job performance [3]."
----

### based on Audio

Speech-based Emotion Recognition (SER) system


Quote from [P11]
------
" A very promising characteristic of DNNs is that they can learn high-level invariant features from raw data [15, 4], which is potentially helpful for emotion recognition."
------


### Personality detection




### Personality types

Research in Psicology suggest that individual's behavior and preferences are conditioned and dependent by underlying psycological constructs, commonly refered as personality traits [3].

Extensive research on approaches how to cluster personality types.

Open questions:
- What are the most accepted personality mapping theories?
- Are there more suitable contexts to apply some theories than others?
- which tests to use?
- Scientific validity of theories??


#### Models

- "Hippocrates and Galen theory - classify the personality into four basic personality types: (1) Sanguine (animated, cheerful, humorist, extrovert, trendsetter);
(2) Choleric (strong, adventurous, powerful, dominant); (3) Melancholy (analytical, individualist, details, planner, perfectionist); (4) Phlegmatic (friendly, easy going, peaceful, shy, adaptable); [10]. These personality types are the most popular and the oldest classification of personalities [11]."


- Myers–Briggs Type Indicator (Form G ??)

- Big Five Personality traits/Five Factor Model (FFM)

The Five Factor Model (FFM) is one of the most accepted models in psicology, and describes constructs personality in five dimensions:  Extraversion, Openness, Conscientiousness, Neuroticism, and Agreeableness [P4]. Research has been conducted accross different cultures in over 50 societies spreading over the six continents [5], supporting the claims of the universality of the model.  

* Openness - 

* Conscientiousness - 
* Extroversion - 
* Agreeableness - 
* Neuroticism/Emotional Stability - 

#### Predicting personality types Methods

Other than using questionaires, there have been several attempts of using other methods of determining personality type, such as hand-writting, TODO

#### Psicology models predictive power

Previous studies suggest that psicology models such as FFM do have indeed predictive power [P5]. For example, individuals with high level of Neuroticism and Openness, as well as low Agreeableness and Conscientiousness demonstrated higher risk of drug use [P5].

Also FFM score and job performance has been subject to study, and promissing results.
New business ventures [P7]



Bibliography

[P1] A. Pauly and D. Sankar, "Novel Online Product Recommendation System Based on Face Recognition and Emotion Detection", 2015

[P2] Golnoosh Farnadi et al., "Computational personality recognition in social media", 2016

[P3] Ozer, D.J., Benet-Martinez, V.: Personality and The Prediction of Consequential Out-
comes. Annual Review of Psychology 57, 401–421 (2006)

[P4] M. Gurven et al., "How Universal Is the Big Five? Testing the Five-Factor Model of Personality Variation Among Forager–Farmers in the Bolivian Amazon", Journal of Personality and Social Psychology, 2013

[P5] E. Fehrman et al., "The Five Factor Model of personality and evaluation of drug consumption risk"

[P6] Sampo V. Paunomen and Michael C. Ashton, "Big Five Factors and Facets and the Prediction of Behavior", Journal of Personality and Social Psicology, 2001  

[P7] "The Use of Personality and the Five-Factor Model to Predict New Business Ventures: From Outplacement to Start-up" (TODO)

[P8] Farnadi, G., Sushmita, S., Sitaraman, G., Ton, N., De Cock, M., Davalos, S., "A Multivariate Regression Approach to Personality Impression Recognition of Vloggers", Proceedings of the WCPR, pp. 1–6, 2014

[P9] R. Hu and P. Pu. "Enhancing collaborative filtering systems with personality information", In Proc. of ACM RecSys, pages 197–204, 2011.

[P10] L. Chen et al,, "An Initial Analysis of Structured Video Interviews by Using Multimodal Emotion Detection", 2014

[P11] "Speech Emotion Recognition Using Deep Neural Network and Extreme
Learning Machine"