# DS4002-Project-1
Career Matcher Project for DS 4002
# DS4002-Project-1
Career Matcher Project for DS 4002
This project develops a career-matching system that uses a person's resume and interests to recommend potential career paths. The system uses occupational information from the O*NET 31.0 Database and Sentence-BERT to compare a user's profile with career profiles and rank careers based on similarity. The goal is for the system to achieve at least 80% top 5 accuracy across the evaluation profiles.  

## Section 1: 
## Software and Platform
This project uses Python for data processing, preprocessing, natural language processing, and analysis. The project is run in Google Colab. The different Python packages to process and analyze the data such as Pandas, which is used to load, organize, and preprocess the O*NET datasets, while NumPy is used for numerical operations and calculating similarity between embeddings. Sentence-transformers is used to generate Sentence-BERT embeddings from career and user profiles. The Sentence-BERT model used in the projedct is sentence-transformers/all-mpet-base-v2. This pretrained model generates 768-dimensional embeddings that are used to compare user profiles with career profiles.
## Platform
The project was developed and analyzed using Google Colab. The project files and documentation are maintained in a google drive folder and a GitHub repository. 

## Section 2: Map of Documentation

DS-4002-Project-1/
|
|--README.md
|--LICENSE.md 
|
|--SCRIPTS/
  |--[preprocessing script]
  |--[embedding script]
  |--[similarity/evaluation script]
|
|--DATA/
  |--occupation_data.csv
  |--task_statements.csv
  |--essential_skills.csv
  |__README.md
|
|__OUTPUT/
  |--[figures]
  |--[tables]
  |--[model outputs]
