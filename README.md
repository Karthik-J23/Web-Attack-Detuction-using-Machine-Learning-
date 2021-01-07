
### Web Attack Detuction Using Machine Learning
## What is web app and web attack?

A web application is a computer program that utilizes web browsers and web technology to perform tasks over the Internet.
Web attacks can be a serious threat to a web application they take advantage on the vulnerability of the application gain access of the application DB and do serious damage.

## Business Constrain:

1)Web attack can be a serious threat to a web application they take advantage on the vulnetrabilty of the application gain acceess of the application DB and do serius damage
2)There are various types of web attacks which attacks the web application and cause serious damage.
3)Existing way to avoid the web attack is to use Intrusion detuction system, which can detuck the mallicious web request and stop them from damaging the web application.
4)But the IDS cannot detuct and restrict the attacks in full scale as they are limited to observe static patterns in the web requests, hence some requests which are mallicious but do not follow the pattern may easily cross the IDS and cause damage to web application

## Objective:

To build an Network based Intrusion detuction system which can detuct the malicious traffic flow over the application layer and stop them from entering into the Web application there by protecting the Web application

## Performance Metrics:

1) True postive and False positive rate can be the important constrains in handling this issue, traffics which are considered to be false negative and passed can cause damage to the application 2) Precision, Recall, F1 score are the Performance metrices accounted here

## Dataset:

The dataset used here is the (CSE-CIC-IDS2018) dataset:https://www.kaggle.com/solarmainframe/ids-intrusion-csv?select=02-14-2018.csv

A infrastructure is collected by building a manual infrastructure of Switches, Servers and PCs and bening and mallicious traffic signals are passed and the same has been collected on each days.
