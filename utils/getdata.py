import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_data(path='./ETT/ETTm2.csv'):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])

    scaler = StandardScaler()
    data = scaler.fit_transform(df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']].values)

    train_data = data[:int(0.6 * len(data)), :]
    valid_data = data[int(0.6 * len(data)):int(0.8 * len(data)), :]
    test_data = data[int(0.8 * len(data)):, :]

    return train_data, valid_data, test_data
