import pandas as _VSCODE_pd
import numpy as _VSCODE_np
import builtins as _VSCODE_builtins
import json as _VSCODE_json
import pandas.io.json as _VSCODE_pd_json

# PyTorch and TensorFlow tensors which can be converted to numpy arrays
_VSCODE_allowedTensorTypes = ["Tensor", "EagerTensor"]

# Function to convert >2D numpy arrays to a flat 2D DataFrame where
# remaining dimensions are represented as strings
def _VSCODE_convertNumpyNdArrayToDataFrame(data, truncate_long_strings):
    try:
        if hasattr(data, "ndim") and data.ndim > 2:
            x_len = data.shape[0]
            y_len = data.shape[1]
            flattened = _VSCODE_np.empty((x_len, y_len), dtype=object)
            # Unfortunately, numpy, pandas and pytorch don't provide an API
            # to accomplish this in native code
            for i in range(x_len):
                for j in range(y_len):
                    if truncate_long_strings:
                        # Generate a preview of the data from the first three elements
                        flat = data[i][j]
                        first_three = flat[:3]
                        currval = _VSCODE_np.array2string(first_three, separator=", ")
                        flattened[i][j] = (
                            currval[:-1] + ", ...]" if len(flat) > 3 else currval
                        )
                        del flat
                        del first_three
                        del currval
                    else:
                        # Untruncated view of the data was requested
                        flattened[i][j] = _VSCODE_np.array2string(
                            data[i][j], separator=", "
                        )
            data = flattened
            del x_len
            del y_len
            del flattened
        return _VSCODE_pd.DataFrame(data)
    except:
        pass
    return data


# Function that converts tensors to DataFrames
def _VSCODE_convertTensorToDataFrame(tensor, truncate_long_strings):
    try:
        temp = tensor
        # Can't directly convert sparse tensors to numpy arrays
        # so first convert them to dense tensors
        if hasattr(temp, "is_sparse") and temp.is_sparse:
            # This guard is needed because to_dense exists on all PyTorch
            # tensors and throws an error if the tensor is already strided
            temp = temp.to_dense()
        # Two step conversion process required to convert tensors to DataFrames
        # tensor --> numpy array --> dataframe
        temp = temp.numpy()
        temp = _VSCODE_convertNumpyNdArrayToDataFrame(temp, truncate_long_strings)
        tensor = temp
        del temp
    except AttributeError:
        # TensorFlow EagerTensors and PyTorch Tensors support numpy()
        # but avoid a crash just in case the current variable doesn't
        pass
    return tensor


# Function that converts the var passed in into a pandas data frame if possible
def _VSCODE_convertToDataFrame(df, truncate_long_strings):
    vartype = type(df)
    if isinstance(df, list):
        df = _VSCODE_pd.DataFrame(df)
    elif isinstance(df, _VSCODE_pd.Series):
        df = _VSCODE_pd.Series.to_frame(df)
    elif isinstance(df, dict):
        df = _VSCODE_pd.Series(df)
        df = _VSCODE_pd.Series.to_frame(df)
    elif hasattr(df, "toPandas"):
        df = df.toPandas()
    elif (
        hasattr(vartype, "__name__") and vartype.__name__ in _VSCODE_allowedTensorTypes
    ):
        df = _VSCODE_convertTensorToDataFrame(df, truncate_long_strings)
    elif hasattr(vartype, "__name__") and vartype.__name__ == "ndarray":
        df = _VSCODE_convertNumpyNdArrayToDataFrame(df, truncate_long_strings)
    else:
        """Disabling bandit warning for try, except, pass. We want to swallow all exceptions here to not crash on
        variable fetching"""
        try:
            temp = _VSCODE_pd.DataFrame(df)
            df = temp
            del temp
        except:  # nosec
            pass
    del vartype
    return df


# Function to compute row count for a value
def _VSCODE_getRowCount(var):
    if hasattr(var, "shape"):
        try:
            # Get a bit more restrictive with exactly what we want to count as a shape, since anything can define it
            if isinstance(var.shape, tuple):
                return var.shape[0]
        except TypeError:
            return 0
    elif hasattr(var, "__len__"):
        try:
            return _VSCODE_builtins.len(var)
        except TypeError:
            return 0


# Function to retrieve a set of rows for a data frame
def _VSCODE_getDataFrameRows(df, start, end, truncate_long_strings=False):
    df = _VSCODE_convertToDataFrame(df, truncate_long_strings)

    # Turn into JSON using pandas. We use pandas because it's about 3 orders of magnitude faster to turn into JSON
    rows = df.iloc[start:end]
    return _VSCODE_pd_json.to_json(None, rows, orient="table", date_format="iso")


# Function to get info on the passed in data frame
def _VSCODE_getDataFrameInfo(df):
    df = _VSCODE_convertToDataFrame(df, False)
    rowCount = _VSCODE_getRowCount(df)

    # If any rows, use pandas json to convert a single row to json. Extract
    # the column names and types from the json so we match what we'll fetch when
    # we ask for all of the rows
    if rowCount:
        try:
            row = df.iloc[0:1]
            json_row = _VSCODE_pd_json.to_json(None, row, date_format="iso")
            columnNames = list(_VSCODE_json.loads(json_row))
        except:
            columnNames = list(df)
    else:
        columnNames = list(df)

    # Compute the index column. It may have been renamed
    try:
        indexColumn = df.index.name if df.index.name else "index"
    except AttributeError:
        indexColumn = "index"

    columnTypes = _VSCODE_builtins.list(df.dtypes)

    # Make sure the index column exists
    if indexColumn not in columnNames:
        columnNames.insert(0, indexColumn)
        columnTypes.insert(0, "int64")

    # Then loop and generate our output json
    columns = []
    for n in _VSCODE_builtins.range(0, _VSCODE_builtins.len(columnNames)):
        column_type = columnTypes[n]
        column_name = str(columnNames[n])
        colobj = {}
        colobj["key"] = column_name
        colobj["name"] = column_name
        colobj["type"] = str(column_type)
        columns.append(colobj)

    # Save this in our target
    target = {}
    target["columns"] = columns
    target["indexColumn"] = indexColumn
    target["rowCount"] = rowCount

    # return our json object as a string
    return _VSCODE_json.dumps(target)
