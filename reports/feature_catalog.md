# Feature Catalog

| Feature | Entity | Type | Description | Source |
|---|---|---|---|---|
| user_avg_rating | user | float | Mean rating given by the user | train interactions |
| user_activity_count | user | int | Number of interactions by the user | train interactions |
| days_since_last_interaction | user | float | Days between the user's last interaction and end of training period | train interactions |
| preferred_genre | user | category | Genre with the most interactions for the user | train + items |
| age, gender, occupation | user | mixed | Demographic attributes | users table |
| popularity | item | int | Number of ratings the item received | train interactions |
| item_avg_rating | item | float | Mean rating the item received | train interactions |
| genre flags (19 columns) | item | binary | One-hot genre indicators | items table |
| rating | interaction | int | Rating given (1-5) | train interactions |
| recency_days | interaction | float | Age of the interaction relative to end of training period | train interactions |

All features are computed from the training set only (no test leakage) by the single function build_features(), which is reused for training and prediction.
