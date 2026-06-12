Dataset required: fake_job_postings.csv
=======================================

This file is NOT included in the repository because it is ~47 MB (exceeds
GitHub's recommended 50 MB soft limit and is best sourced directly from Kaggle).

Download it here:
  https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

Place the downloaded file in this directory as:
  data/fake_job_postings.csv

Then run:
  python train.py

The SQLite database (predictions.db) is also created in this directory
automatically when you first run app.py.
