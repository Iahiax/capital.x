# scenario_engine.py

def filter_scenario(df, scenario_type):
    if scenario_type == 'trend':
        return df[df['Regime'].isin([1, -1])]
    if scenario_type == 'range':
        return df[df['Regime'] == 0]
    if scenario_type == 'chaos':
        return df[df['Regime'] == 2]
    return df
