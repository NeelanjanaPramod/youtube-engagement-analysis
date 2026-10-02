# YouTube Engagement Analysis: predicting likes

Exploratory analysis and regression modeling on US YouTube trending videos. Goal: predict a video's like count from its other stats and channel history.

## Dataset
- Kaggle "YouTube Trending Video Dataset" (US): `2-US_youtube_trending_data.csv.gz` plus `1-US_category_id.json`. The data files are not stored in this repo (6 MB gz; download them from Kaggle).
- 199,190 rows, 0 nulls, 36,279 unique videos. Rows are grouped by `video_id` (max values) before modeling.

## Pipeline
1. Load data, map category ids to genres, group by video.
2. EDA: genre breakdowns (Music leads with 28.5B views), treemap, year trends, top channels.
3. Baselines on a 75/25 split predicting likes: Linear Regression 0.797, Random Forest 0.866, XGBoost 0.676.
4. Tuning (RandomizedSearch, 3-fold): Ridge 0.797, RF 0.857, XGB 0.735. No gain over defaults for RF.
5. Feature engineering: log views/comments/dislikes, comments and dislikes per view, days from publish to trending, publish hour/day, per-channel video count, and a leak-free channel like-rate (computed out-of-fold on train only).
6. Target transform: predict log like-rate (likes/views), multiply back by views.
7. Final models, scored on the held-out test set.

## Results (R2, held-out test, raw likes)
| Model | R2 |
|---|---|
| Random Forest | 0.915 |
| LightGBM | 0.903 |
| XGBoost | 0.902 |
| Average of 3 | 0.911 |
| Linear Regression (baseline) | ~0.80 |

Error flattens around 0.91, so higher scores were not reachable honestly on this data.

## Run
Open `Data_Analysis.ipynb` in Google Colab, Runtime > Run all, and upload the two Kaggle files when prompted. `final_pipeline.py` is the final pipeline as a plain script.
