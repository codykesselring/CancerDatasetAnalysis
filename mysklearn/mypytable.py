from mysklearn import myutils

# TODO: copy your mypytable.py solution from PA2-PA6 here
from mysklearn import myutils
import copy
import csv
from tabulate import tabulate
import os

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names (list of str): M column names
        data (list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Parameters:
            column_names (list of str): initial M column names (None if empty)
            data (list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure."""
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            tuple: (N, M) where N is number of rows and M is number of columns
        """
        return len(self.data), len(self.data[0]) # TODO: fix this

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Parameters:
            col_identifier (str or int): string for a column name or int
                for a column index
            include_missing_values (bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Raises:
            ValueError: if col_identifier is invalid
        """
        try:
            index = self.column_names.index(col_identifier)
            for row in self.data:
                return [row[index] for row in self.data]

        except ValueError as e:
            raise print(f"Invalid column identifier {e}") 
        
        return [] # TODO: fix this

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leaves values as-is that cannot be converted to numeric.
        """
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                try:
                    self.data[i][j] = float(self.data[i][j])
                except (ValueError, TypeError):
                    # Do nothing if conversion fails (e.g., if it's "NA" or a string)
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Parameters:
            row_indexes_to_drop (list of int): list of row indexes to remove from the table data.
        """
        for index in sorted(row_indexes_to_drop, reverse=True):
            if 0 <= index < len(self.data):
                del self.data[index]

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: returns self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Uses the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load.
        """
        
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            
            self.column_names = next(reader)
            
            for row in reader:
                self.data.append(row)
        
        self.convert_to_numeric()
        
        return self


    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to save the contents to.

        Notes:
            Uses the csv module.
        """
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.column_names)
            writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Parameters:
            key_column_names (list of str): column names to use as row keys.

        Returns:
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
            The first instance of a row is not considered a duplicate.
        """
        seen_rows = set()
        duplicate_indexes = []

        key_indexes = [self.column_names.index(col) for col in key_column_names]

        for i, row in enumerate(self.data):
            key_values = tuple(row[index] for index in key_indexes)
            if key_values in seen_rows:
                duplicate_indexes.append(i)
            else:
                seen_rows.add(key_values)

        return duplicate_indexes

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA")."""
        new_data = []
        for row in self.data:
            if 'NA' not in row:
                new_data.append(row)
        self.data = new_data

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
        by the column's original average.

        Parameters:
            col_name (str): name of column to fill with the original average (of the column).
        """
        avg = self.compute_summary_statistics([col_name]).data[0][4]
        for row in self.data:
            index = self.column_names.index(col_name)
            if row[index] == 'NA':
                row[index] = avg

   

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Parameters:
            col_names (list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values in the columns to compute summary stats
            should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        statistics = []
        for name in col_names:
            values = []
            index = self.column_names.index(name)
            for row in self.data:
                val = row[index]
                if val != "NA":
                    values.append(val)
            if not values:
                return MyPyTable(
                    column_names=["attribute", "min", "max", "mid", "avg", "median"],
                    data=[])
            values.sort()
            min_val = min(values)
            max_val = max(values)
            avg_val = sum(values) / len(values)
            mid_val = (min_val + max_val) / 2
            if len(values) % 2 == 0:
                median_val = (values[len(values)//2 - 1] + values[len(values)//2]) / 2
            else:
                median_val = values[len(values)//2]
            statistics.append([name, min_val, max_val, mid_val, avg_val, median_val])
        return MyPyTable(
            column_names=["attribute", "min", "max", "mid", "avg", "median"],
            data=statistics)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
        with other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.
            
        Returns:
            MyPyTable: the inner joined table.
        """
        new_data = []

        self_key_indexes = [self.column_names.index(col) for col in key_column_names]
        other_key_indexes = [other_table.column_names.index(col) for col in key_column_names]

        other_non_key_indexes = [
            i for i, col in enumerate(other_table.column_names)
            if col not in key_column_names
        ]

        for self_row in self.data:
            self_key = tuple(self_row[index] for index in self_key_indexes)
            for other_row in other_table.data:
                other_key = tuple(other_row[index] for index in other_key_indexes)
                if self_key == other_key:
                    combined_row = self_row + [other_row[i] for i in other_non_key_indexes]
                    new_data.append(combined_row)

        new_column_names = self.column_names + [
            col for col in other_table.column_names if col not in key_column_names
        ]
        return MyPyTable(new_column_names, new_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
        other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pads attributes with missing values with "NA".
        """
        new_data = []

        self_key_indexes = [self.column_names.index(col) for col in key_column_names]
        other_key_indexes = [other_table.column_names.index(col) for col in key_column_names]

        #stores indexes of non-key columns in other_table
        other_non_key_indexes = [
            i for i, col in enumerate(other_table.column_names)
            if col not in key_column_names
        ]

        self_key_to_rows = {}
        other_key_to_rows = {}

        for row in self.data:
            key = tuple(row[idx] for idx in self_key_indexes)
            if key not in self_key_to_rows:
                self_key_to_rows[key] = []
            self_key_to_rows[key].append(row)

        for row in other_table.data:
            key = tuple(row[idx] for idx in other_key_indexes)
            if key not in other_key_to_rows:
                other_key_to_rows[key] = []
            other_key_to_rows[key].append(row)

        # Get all unique keys
        all_keys = set(self_key_to_rows.keys()) | set(other_key_to_rows.keys())

        for key in all_keys:
            self_rows = self_key_to_rows.get(key, [])
            other_rows = other_key_to_rows.get(key, [])

            if self_rows and other_rows:
                for self_row in self_rows:
                    for other_row in other_rows:
                        combined_row = self_row + [other_row[i] for i in other_non_key_indexes]
                        new_data.append(combined_row)
            elif self_rows:
                for self_row in self_rows:
                    combined_row = self_row + ["NA"] * len(other_non_key_indexes)
                    new_data.append(combined_row)
            else:
                for other_row in other_rows:
                    padded_self = ["NA"] * len(self.column_names)
                    for i, idx in enumerate(self_key_indexes):
                        padded_self[idx] = other_row[other_key_indexes[i]]
                    combined_row = padded_self + [other_row[i] for i in other_non_key_indexes]
                    new_data.append(combined_row)

        new_column_names = self.column_names + [
            col for col in other_table.column_names if col not in key_column_names
        ]
        return MyPyTable(new_column_names, new_data) # Return empty table on error