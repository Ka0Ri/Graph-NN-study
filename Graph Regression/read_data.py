import re
import numpy as np
import pandas as pd


def read_sample(df, min_len = 60, input_length = 10, out_put_length=1, step = 1, with_label=False):
    data = []
    label = []
    next_sample = []
   
    for columns_name in df.columns:
        columns = df[columns_name][df[columns_name].notna()]
        list_data = [float(re.sub('[.,]', "", str(item))) for item in columns.to_list()]
        if(len(list_data) < min_len):
            continue
        else:
            for i in range(0, len(list_data) - input_length - 1, step):
                data.append(list_data[i:i+input_length])
                next_sample.append(list_data[(i+input_length-out_put_length+1):(i+input_length+1)])
                label.append(columns_name)
    if(with_label == True):
        return np.array(data), np.array(next_sample), label
    else:
        return np.array(data), np.array(next_sample)


def read_all_data(df, min_len=60, step=1):

    data = []
    label = []
    for columns_name in df.columns:
        columns = df[columns_name][df[columns_name].notna()]
        list_data = [float(re.sub('[.,]', "", str(item))) for item in columns.to_list()]
        if(len(list_data) < min_len):
            continue
        else:
            data.append(list_data)
            label.append(columns_name)

    return data, label


if __name__ == "__main__":

    xls_data = pd.read_excel('sample_data.xlsx')
    data, label = read_all_data(xls_data)

    print(data[0])
    print(label[0])
    