# AI Recommendation Logic - Project 3

A clean, explainable recommendation system built for the DecodeLabs Artificial Intelligence Project 3 brief.

## What the project does

The system accepts a user's interests and optional preferences for difficulty, content format, and learning style. It then compares the user profile with a catalog of AI/tech learning resources and ranks the best matches.

This directly covers the required project flow:

- Take user input (choices/interests)
- Match preferences using logic/similarity
- Display recommended items

## Engineering approach

This is a **content-based recommendation system**. It does not randomly suggest items and it does not need historical user-rating data.

Each user and resource is converted into a weighted feature vector:

- Topic / interest match: weight 3.0
- Difficulty match: weight 1.5
- Format match: weight 1.0
- Learning-style match: weight 1.0

The engine calculates **cosine similarity** between the user vector and every resource vector:

`cosine_similarity = (A · B) / (||A|| × ||B||)`

A very small rating bonus is used only to break close ties. The system also returns human-readable reasons for each recommendation, making the result explainable.

## Project structure

```text
AI_Recommendation_Logic_Project3/
├── app.py                     # Streamlit web interface
├── main.py                    # Command-line interface
├── recommender.py             # Core recommendation engine
├── requirements.txt
├── data/
│   └── resources.csv          # Sample item catalog
└── tests/
    └── test_recommender.py    # Unit tests
```

## Run the command-line version

Requires Python 3.10+.

```bash
python main.py
```

The CLI itself uses only the Python standard library.

## Run the web interface

Install Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run tests

From the project folder:

```bash
python -m unittest discover -s tests -v
```

## Example user profile

- Interests: `python, machine learning, recommendation systems`
- Difficulty: `intermediate`
- Format: `project`
- Learning style: `hands-on`

The engine should prioritize resources such as **Content-Based Filtering Project** because its attributes strongly overlap with the profile.

## Why this design is suitable for Project 3

The goal of this milestone is preference mapping and pattern alignment before more advanced collaborative-filtering or neural recommendation models. This implementation intentionally focuses on transparent similarity logic, while keeping the code modular enough to upgrade later.

## Possible future upgrades

- Learn preference weights from explicit user ratings
- Add TF-IDF embeddings for free-text descriptions
- Add collaborative filtering once user-item interaction data exists
- Add hybrid content + collaborative ranking
- Persist users and interaction history in a database
- Add evaluation metrics such as Precision@K and Recall@K
- Deploy the Streamlit app online
