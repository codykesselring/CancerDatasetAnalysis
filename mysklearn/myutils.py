# TODO: your reusable general-purpose functions here
# TODO: your reusable general-purpose functions here
import numpy as np

def get_column(table, header, col_name):
    col_index = header.index(col_name)
    col = []
    for row in table:
        col.append(row[col_index])

    return col

def get_frequencies(table, header, col_name):
    # TODO: resolve this using python dictionaries
    col = get_column(table, header, col_name)
    # we want to get the unique values from this column
    unique_col_values = sorted(list(set(col)))
    #print(unique_col_values)
    counts = []
    for val in unique_col_values:
        counts.append(col.count(val))

    return unique_col_values, counts

def group_by(table, header, group_by_col_name):
    # returns the grouped rows in a structure that makes it easy to access each group
    #(for example, a dictionary with the attribute values as keys, or a list of tables with corresponding labels).
    
    col_idx=header.index(group_by_col_name)
    subtables={}

    for row in table:
        key=row[col_idx]
        if key not in subtables:
            subtables[key]=[]
        subtables[key].append(row)
    return subtables
        
    
    
# practice question solution from Discretization 
def compute_equal_width_cutoffs(values, num_bins):
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins

    # generate cutoffs
    cutoffs = [min(values) + i*bin_width for i in range(num_bins)]
    # append the maximum value
    cutoffs.append(max(values))
    # optionally, round
    cutoffs = [round(cutoff,2) for cutoff in cutoffs]
    return cutoffs

def compute_bin_frequencies(values, cutoffs):
    freqs = [0 for _ in range(len(cutoffs) - 1)] # because N + 1 cutoffs

    for value in values:
        if value == max(values):
            freqs[-1] += 1 # add one to the last bin count
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= value < cutoffs[i + 1]:
                    freqs[i] += 1  # add one to this bin defined by [cutoffs[i], cutoffs[i+1])
    return freqs

def compute_slope_intercept(x, y):
    meanx = sum(x) / len(x)
    meany = sum(y) / len(y)

    num = sum([(x[i] - meanx) * (y[i] - meany) for i in range(len(x))])
    den = sum([(x[i] - meanx) ** 2 for i in range(len(x))])
    m = num / den
    # y = mx + b -> y - mx
    b = meany - m * meanx
    return m, b
