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

## Section 3: Instructions for Reproducing Results

Step 1: Open project notebook in Google Colab. 

Step 2: Run the notebook from the beginning in order with datasets (occupation_data.csv, task_statements.csv, essential_skills.csv).

Step 3: The notebook will use the Sentence-BERT tokenizer aspect to check the length of each career profile. Profiles that exceed the 384-token limit will be shortened using the projects preprocessing rules. 

Step 4: By the end, the notebook will have created the final career-profile dataset career_profiles.csv and will be used for the input career-matching model .

Step 5: The career profiles will then be used in the Sentence-Bert model. The model will convert each career profile into a numerical embedding. 

Step 6: A user's resume and interests will be combined into one profile and converted into an embedding using the same Sentence-BERT model. The user embedding will then be compared with the career embeddings using cosine similarity. The careers will be ranked from highest to lowest similarity, and the top five career matches will be returned for each user profile. 

Step 7: The recommendations will be evaluated using Top 5 accuracy, which measures whether at least one expected career appears among the five recommended careers. 

