import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import tabulate

# unix datetime
base = pd.Timestamp("1970-01-01")
CHUNK_SIZE = 1000000
REVIEW_USER_DROP = 100
REVIEW_RESTAURANT_DROP = 100
ROWS_LIMIT = int(1e3)
RESTAURANTS_PATH = 'data/yelp_dataset/yelp_academic_dataset_business.json'
REVIEWS_PATH = 'data/yelp_dataset/yelp_academic_dataset_review.json'
USERS_PATH = 'data/yelp_dataset/yelp_academic_dataset_user.json'

# Load data files
# reviews = get_reviews()

def plot_data():
    print("Reading users")

    with open(USERS_PATH, 'r', encoding='utf-8') as file:
        # json_lines = file.readlines()  # read full file
        json_data = []
        for _ in range(ROWS_LIMIT):
            try:
                line = next(file)
                data = json.loads(line)
                json_data.append(data)
            except StopIteration:
                # Handle the end of the file
                break

    users = pd.DataFrame(json_data)

    # users = pd.read_csv(USERS_PATH)
    users = users[users['review_count'] > REVIEW_USER_DROP]
    users['user_id'] = users['user_id'].astype('category')
    users['user_id_num'] = users['user_id'].cat.codes
    users = users[['user_id', 'user_id_num', 'review_count']]
    user_id_to_num = dict(zip(users['user_id'], users['user_id_num']))

    print("Reading businesses")

    with open(RESTAURANTS_PATH, 'r', encoding='utf-8') as file:
        # json_lines = file.readlines()  # read full file
        json_data = []
        for _ in range(ROWS_LIMIT):
            try:
                line = next(file)
                data = json.loads(line)
                json_data.append(data)
            except StopIteration:
                # Handle the end of the file
                break

    businesses = pd.DataFrame(json_data)

    main_categories = ['Restaurants', 'Health & Medical', 'Shopping', 'Hotels & Travel']
    businesses = businesses.dropna(subset=['categories'])
    for main_category in main_categories:
        businesses[main_category] = businesses['categories'].apply(lambda x: 1 if main_category in x else 0)

    print("Reading reviews")

    with open(REVIEWS_PATH, 'r', encoding='utf-8') as file:
        # json_lines = file.readlines()  # read full file
        json_data = []
        for _ in range(ROWS_LIMIT):
            try:
                line = next(file)
                data = json.loads(line)
                json_data.append(data)
            except StopIteration:
                # Handle the end of the file
                break

    reviews = pd.DataFrame(json_data)

    # reviews = pd.read_json(REVIEWS_PATH)

    reviews = pd.merge(reviews, users, how='inner', on='user_id')
    reviews = reviews.drop(columns='user_id')
    reviews = pd.merge(reviews, businesses, how='inner', on='business_id')
    reviews = reviews.drop(columns='business_id')
    print("REVIEWS.HEAD() -------------------------------------------------------------------")
    print(reviews.head())
    reviews = reviews.drop(columns=reviews.columns[0], axis=1)
    print("REVIEWS.DROP() -------------------------------------------------------------------")
    print(reviews.head())

    # Create a directory to save the plots if it doesn't exist
    save_directory = 'plots'
    os.makedirs(save_directory, exist_ok=True)

    # # 1) Main Categories Distribution
    # plt.figure(figsize=(8, 6))
    # main_categories_counts = businesses[main_categories].sum()
    # main_categories_counts.plot(kind='bar')
    # plt.title('Main Categories Distribution')
    # plt.xticks(rotation=45)
    # plt.xlabel('Categories')
    # plt.ylabel('Count')
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_directory, 'main_categories_distribution.png'))
    # plt.close()
    #
    # # 2) Rating Distribution for Each Category - Combined Plot
    # plt.figure(figsize=(18, 12))
    # for i, main_category in enumerate(main_categories, 1):
    #     plt.subplot(2, 2, i)
    #     # Check the actual column names in your reviews DataFrame
    #     sns.histplot(data=reviews[reviews[main_category] == 1], x='stars_x', bins=30, kde=True, label=main_category, alpha=0.7)
    #     plt.title(f'Rating Distribution for {main_category}', fontsize=14)
    #     plt.xlabel('Rating', fontsize=13)
    #     plt.ylabel('Count', fontsize=13)
    #     plt.legend(fontsize=16)
    #
    # plt.tight_layout()  # Automatically adjust layout
    # plt.savefig(os.path.join(save_directory, 'combined_rating_distribution.png'))
    # plt.close()

    # 3) Table with Statistics
    stats_table = pd.DataFrame({
        'Main Category': main_categories,
        'Number of Businesses': [businesses[main_category].sum() for main_category in main_categories],
        'Number of Reviews': [reviews[reviews[main_category] == 1].shape[0] for main_category in main_categories]
    })

    table_str = tabulate.tabulate(stats_table, headers='keys', tablefmt='latex', showindex=False)

    # Save the table to a .tex file
    with open(os.path.join(save_directory, 'statistics_by_main_category.tex'), 'w') as f:
        f.write(table_str)

    return

if __name__ == "__main__":
    plot_data()
