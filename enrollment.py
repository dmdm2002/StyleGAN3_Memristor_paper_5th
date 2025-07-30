import os

import pandas
import random
import numpy as np
from scipy.stats import norm


def proportion_from_binary_stream(bit_list, chunk_size=10):
    print(f'Input Size: {len(bit_list)}-bit')
    proportions = []
    for i in range(0, len(bit_list), chunk_size):
        chunk = bit_list[i: i+chunk_size]
        bit_str = ''.join(map(str, chunk))
        decimal_value = int(bit_str, 2)
        proportion = decimal_value / ((2 ** chunk_size) - 1)
        print(f'{bit_str}: {decimal_value} --> {proportion}')
        # chunk = map(int, chunk)
        # p = sum(chunk) / chunk_size
        proportions.append(proportion)

    return proportions


if __name__ == '__main__':
    path = 'raw_data/12X12ARRAY_GAN.xlsx'

    os.makedirs('noise_data/prob_max_new/', exist_ok=True)
    # os.makedirs('./noise_data/gaussian_prob/', exist_ok=True)

    excel_data = pandas.read_excel(path)
    # binary_bit_map = np.array(excel_data['Bit_Output'].tolist()) # 1, 400
    binary_bit_map = np.array(excel_data['Response'].tolist())  # 1, 400
    new_binary_bit_map = []
    for bit_map in binary_bit_map:
        temp = []
        for i in range(len(bit_map)):
            temp += bit_map[i]
            if (i+1) % 100 == 0:
                new_binary_bit_map.append(temp)
                temp = []

    # print(new_binary_bit_map)
    random.shuffle(new_binary_bit_map)
    print(f"Binary Bit Map: {new_binary_bit_map}, Len: {len(new_binary_bit_map)}")
    min = 1e-6
    max = 1 - min

    proportions = []
    noise_name_number = 0
    for i, bit_map in enumerate(new_binary_bit_map):
        proportions += proportion_from_binary_stream(list(bit_map))
        # print(len(proportions))
        if (i+1) % 10 == 0:
            proportions = np.array(proportions)
            clipped_props = np.clip(proportions, min, max)
            np.save(f'./noise_data/prob_max_new/noise_{noise_name_number}', clipped_props)

        #
        #     # 저장: Normalized proportion (uniform [0,1])
        #     np.save(f'./noise_data/normal_new/noise_{noise_name_number}', proportions)
        #
        #     # 정규분포로 변환 (ppf 사용) + clipping 안정성 확보
        #     clipped_props = np.clip(proportions, min, max)
        #     gaussian_noise = norm.ppf(clipped_props)
        #
        #     # 저장: 정규분포 noise
        #     np.save(f'./noise_data/gaussian_new/noise_{noise_name_number}', gaussian_noise)
        #
            proportions = []
            noise_name_number += 1

