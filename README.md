# Comparative Data Analysis Report
## Language as Data Course
Author: Mohammed Majeed

## Project Overview
In a world where language and cultural nuances are vital aspects of communication, the statement "translated talks sound different" can be controversial when viewed through certain lenses. Some people might argue that translation, when done skillfully, can preserve the essence and emotional tone of the original conversation. They might believe that skilled translators can bridge the gap between languages effectively, minimizing significant differences in how the translated version sounds. On the other hand, critics could argue that no matter how skilled a translator is, there are always subtle nuances and cultural references that cannot be fully captured, leading to a loss of authenticity in the translated talk. They might claim that such differences in tone and meaning could potentially lead to misunderstandings or misinterpretations, which can be particularly sensitive in contexts such as diplomatic relations or legal proceedings.


## Data Source
I downloaded the dataset from the official IWSLT 2018 website (https://wit3.fbk.eu/2018-01-b) as part of the Low Resource MT track on TED Talks. 
After downloading the dataset, I unzipped the folder and extracted only the Arabic and English talks for further analysis saved in original_data folder. 
From the available talks, I selected a subset consisting of 300 identical talks from both languages for my research purposes, splitted to train and test as discussed in Assignment 1

## Getting Started
### Prerequisites
Python version 3.9.19 or higher.
Required Python packages are listed in requirements.txt.

### Installing
Please execute the following command to install all the required packages: pip install -r requirements.txt.

### Running the Analysis 
Data Preparation: Run get_all_documents.py to prepare the data for analysis.
Data Analysis on test set : Execute run_all_analyses.py to perform the comparative analysis of the test datasets. If you want to run on the training dataset you need to give the 'run' function and argument 'train' instead of 'test'


## Acknowledgements and Licensing
This project utilizes data from TED Talks, which are made available under the Creative Commons BY-NC-ND license. Full details can be found at TED's official usage policy page: https://www.ted.com/about/our-organization/our-policies-terms/ted-talks-usage-policy.

### Reference Paper
M. Cettolo, C. Girardi, and M. Federico. 2012. "WIT3: Web Inventory of Transcribed and Translated Talks." In Proc. of EAMT, pp. 261-268, Trento, Italy.