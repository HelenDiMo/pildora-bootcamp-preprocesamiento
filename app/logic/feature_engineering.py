def add_bill_ratio(df):
    df['bill_ratio'] = df['bill_length_mm'] / df['bill_depth_mm']
    return df
