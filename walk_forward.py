# walk_forward.py

import pandas as pd

def walk_forward(df, train_window='90D', test_window='30D', trainer_fn=None, eval_fn=None):
    results = []
    start = df.index.min()
    end = df.index.max()

    current = start
    while current + pd.Timedelta(train_window) + pd.Timedelta(test_window) < end:
        train_end = current + pd.Timedelta(train_window)
        test_end = train_end + pd.Timedelta(test_window)

        df_train = df[(df.index >= current) & (df.index < train_end)]
        df_test  = df[(df.index >= train_end) & (df.index < test_end)]

        model = trainer_fn(df_train)
        stats = eval_fn(model, df_test)
        results.append(stats)

        current = train_end

    return results
