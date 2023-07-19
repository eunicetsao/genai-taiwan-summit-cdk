from uuid import uuid4

import pandas as pd

data = []
import random
import datetime

import os

current_folder = os.path.dirname(os.path.abspath(__file__))


def random_date(start, end):
    return start + datetime.timedelta(
        seconds=random.randint(0, int((end - start).total_seconds())))


start = datetime.datetime(2022, 9, 1)
end = datetime.datetime(2022, 12, 1)


def generate_500_ids():
    user_ids = []
    for i in range(500):
        user_ids.append(uuid4())

    return user_ids


users = generate_500_ids()

product_lists = {
    'Fruits': 15.5,
    'Milk': 21.2,
    'Chips': 6.3,
    'Shampoo': 55,
    'Ice cream': 30
}

for i in range(10000):
    product, price = random.choice(list(product_lists.items()))
    temp = {
        'transaction_date': random_date(start, end).strftime("%Y-%m-%d"),
        'user_id': random.choice(users),
        'product': product,
        'price': price,

    }

    data.append(temp)
df = pd.DataFrame(data)

print(os.path.join(current_folder, '../samples/data/retail.csv'))
df.to_csv(os.path.join(current_folder, '../samples/data/retail.csv'), index=False, header=None)
