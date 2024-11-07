import datetime
from typing import Callable, Literal, Iterable
import typing
import holidays
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import sklearn.preprocessing
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import torch.utils.data as data

# Author: Antonio Oladuti
# Min Python version: 3.10
plt.style.use("dark_background")

TRADING_MONTH = 22
UK_hols = US_hols = CH_hols = None # lazy init
MOVE_COL_EXT = "_UP"

LOGGING = ["ALL", "FATAL", "ERROR", "WARN", "INFO", "DEBUG"]

AGG_FUNC_DICT = {
    "mean": pd.DataFrame.mean,
    "sum": pd.DataFrame.sum,
    "min": pd.DataFrame.min,
    "max": pd.DataFrame.max,
    "count": pd.DataFrame.count,
    "std": pd.DataFrame.std,
    "var": pd.DataFrame.var,
    "median": pd.DataFrame.median,
    "prod": pd.DataFrame.prod,
    "sem": pd.DataFrame.sem,
    "skew": pd.DataFrame.skew,
    "kurt": pd.DataFrame.kurt,
    "quantile": pd.DataFrame.quantile,
    "nunique": pd.DataFrame.nunique,
    "idxmin": pd.DataFrame.idxmin,
    "idxmax": pd.DataFrame.idxmax,
    "first": pd.DataFrame.first,
    "last": pd.DataFrame.last,
    "all": pd.DataFrame.all,
    "any": pd.DataFrame.any }

# LSTM inputs a 3D tensor: [samples, time steps, features]

class Magic(nn.Module):
    """ Basic LSTM neural network """
    def __init__(
            self, 
            n_features: int, 
            num_layers: int = 1, 
            hidden_size: int = 50):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True)
        self.input_size = n_features
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.linear = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

def adam(model_or_params: nn.Module | Iterable[nn.Parameter]
         ) -> torch.optim.Adam:
    """Return a basic Adam optimizer with default settings.

    Args:
        model_or_params (Module | Iterable[Parameter]): 
            a PyTorch model or its parameters

    Returns:
        Adam: a new Adam optimizer with default settings
    """
    if isinstance(model_or_params, nn.Module):
        return torch.optim.Adam(model_or_params.parameters())
    else:
        return torch.optim.Adam(model_or_params)

def _prep_feature_cols(
        df: pd.DataFrame, 
        feature_cols: list | None, 
        target_col_str: str) -> list:
    if feature_cols is not None:
        fcstrs = resolve_col_names(df, feature_cols)
    if feature_cols is not None and target_col_str not in fcstrs:
        return feature_cols + [target_col_str]
    elif feature_cols is None:
        return df.columns.to_list()
    else:
        return feature_cols

# vectorize and broadcast
def _vect(obj, length: int, np_as_int: bool = True) -> list | np.ndarray:
    if isinstance(obj, list):
        if len(obj) != length and len(obj) != 1:
            raise ValueError("List object must be length 1 or n_models")
        return [obj[0]] * length
    elif isinstance(obj, np.ndarray):
        if np_as_int:
            return np.broadcast_to(obj, length).astype(int)
        else:
            return np.broadcast_to(obj, length)
    else:
        return [obj] * length
    
def df_new_historical_split(
        df: pd.DataFrame, 
        steps_into_past: int, 
        steps_back_to_see: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into [A, B]. A: a DataFrame that goes back 
    to `steps_into_past` time steps in history. 
    B: a DataFrame that starts at `steps_back_to_see` and goes
    forward to the final time step in df.

    Args:
        df (pd.DataFrame): DataFrame to split
        steps_into_past (int): steps back to end the dataframe A
        steps_back_to_see (int): steps back to start the dataframe 

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: [A, B]
    """
    ret1 = df.iloc[:-steps_into_past] if steps_into_past > 0 else df
    if steps_back_to_see > 0:
        ret2 = df.tail(steps_back_to_see)
    else:
        ret2 =  pd.DataFrame(columns=df.columns)
    return (ret1, ret2)

def _remove_holidays(
        date_range: pd.DatetimeIndex, 
        holiday_base: holidays.HolidayBase) -> list[pd.Timestamp]:
    drange_list = []
    for i in range(len(date_range)):
        if date_range[i].date() not in holiday_base:
            drange_list.append(date_range[i])
    return drange_list
    
def _make_future_date_list(
        start_date: pd.DataFrame,
        n_dates: int,
        ignore_uk_hols: bool,
        ignore_us_hols: bool,
        ignore_ch_hols: bool,
        freq: str = "D",
        date_range_list: list | None = None,
        safety_mult: int = 3) -> list:
    if date_range_list is None:
        drange = pd.date_range(
            start=start_date + datetime.timedelta(days=1),
            periods=n_dates * safety_mult,
            freq=freq)
        ret = None
        if ignore_uk_hols:
            global UK_hols
            if UK_hols is None:
                UK_hols = holidays.country_holidays("UK", language="uk")
            ret = _remove_holidays(drange, UK_hols)
        if ignore_us_hols:
            global US_hols
            if US_hols is None:
                US_hols = holidays.country_holidays("US", language="uk")
            ret = _remove_holidays(drange, US_hols)
        if ignore_ch_hols:
            global CH_hols
            if CH_hols is None:
                CH_hols = holidays.country_holidays("CH", language="uk")            
            ret = _remove_holidays(drange, CH_hols)
        if ret is None:
            return drange[:n_dates].tolist()
        else:
            return ret[:n_dates]
    else:
        return date_range_list[:n_dates]

def make_future_date_list(
        df: pd.DataFrame,
        n_dates: int,
        ignore_hols: list[Literal["UK", "US", "CH"]] = [],
        business_days_only: bool = False,
        date_range_list: list | None = None,
        safety_mult: int = 3) -> list: # safety_mult ensures list is n_dates
    """Generate a date list for the future.

    Args:
        df (pd.DataFrame): DataFrame with an earliest-to-latest Date index
        n_dates (int): 
            Number of future dates to return \
            (returned list has length n_dates)
        ignore_hols (list[Literal["UK", "US", "CH"]], optional):
            Certain countries' holiday dates to ignore. Defaults to [].
        business_days_only (bool, optional):
            If True, only includes business days in the returned list. \
            Defaults to False.
        date_range_list (list | None, optional): 
            If not None, this function returns date_range_list[:n_dates]. 
            Defaults to None.
        safety_mult (int, optional): Argument to ensure date range is covered.
            Multiply n_dates by safety_mult to ensure. Defaults to 3.

    Returns:
        list: future date list
    """
    ignore_hols_upper = [s.upper() for s in ignore_hols]
    return _make_future_date_list(
        df.index[-1], 
        n_dates, 
        "UK" in ignore_hols_upper,
        "US" in ignore_hols_upper, 
        "CH" in ignore_hols_upper,
        "B" if business_days_only else "D", 
        date_range_list, 
        safety_mult)
    
def resolve_col_name(df: pd.DataFrame, col_index_or_name) -> str:
    """Return the column name associated with 
    `col_index_or_name` in `df` as a string. 

    Args:
        df (pd.DataFrame): DataFrame
        col_index_or_name: column identifier

    Returns:
        str: returned column name
    """
    if type(col_index_or_name) == str:
        return col_index_or_name
    elif type(col_index_or_name) == int:
        dt_col = df.iloc[:, col_index_or_name]
    else:
        dt_col = df[col_index_or_name]
    return dt_col.name

def resolve_col_names(
        df: pd.DataFrame, 
        cols: list | pd.Index | str | int | typing.Any) -> list[str]:
    """Get the name or names of `cols` in `df` as a list of strings. 

    Args:
        df (pd.DataFrame): DataFrame
        cols (list | pd.Index | str | int): column identifiers

    Returns:
        list[str] | str: returned column names
    """
    if isinstance(cols, pd.Index):
        _cols = cols.to_list()
    else:
        _cols = cols
    if isinstance(_cols, (list, np.ndarray)):
        ret = []
        for i in range(len(_cols)):
            ret.append(resolve_col_name(df, _cols[i]))
        return ret
    else:
        return [resolve_col_name(df, _cols)]

def fill_nan_numerics(df: pd.DataFrame, val: int | float = 0) -> pd.DataFrame:
    """Return a `df` copy with the NaN values in numeric columns \
       set to `val`.

    Args:
        df (pd.DataFrame): DataFrame
        val (int | float, optional):
            value to replace NaN entries with. Defaults to 0.

    Returns:
        pd.DataFrame: DataFrame with NaN values set to `val`
    """
    _df = df.copy()
    numeric_columns = _df.select_dtypes(include=["number"]).columns
    _df.loc[:, numeric_columns] = _df.loc[:, numeric_columns].fillna(val)
    return _df

def train_xy_split(
        data: torch.Tensor, 
        lookback: int = 60, 
        sliding_window_divisor: int | None = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Take training data and create the X (samples) as a 3D tensor of shape
    (n_samples, lookback window, n_features). Create y (labels) as 
    a 2D array of shape (n_samples, n_features). Return tuple (X, y).

    Args:
        data (torch.tensor): 2D tensor of training data
        lookback (int, optional): 
            number of time steps stored in each X sample. 
            Defaults to 60.
        sliding_window_divisor (int | None, optional):
            Divide the `lookback` by `sliding_window_divisor`
            and jump (`lookback // sliding_window_divisor`) time steps 
            per X and y sample entry in training. 
            If None, no adjustment is made, and the window slides by 1
            time step. Defaults to None.

    Returns:
        tuple[torch.Tensor,torch.tensor]: (X, y)
    """
    X, y = [], []
    if sliding_window_divisor is None:
        jumper = lookback
    else:
        jumper = lookback // sliding_window_divisor
    for i in range(lookback, len(data), jumper):
        X.append(data[i - lookback : i, :])
        y.append(data[i])
    X_tensor = torch.stack(X).to(torch.float32)
    return X_tensor, torch.stack(y).reshape(
        X_tensor.shape[0], X_tensor.shape[2]).to(torch.float32)

def gen_moves(
        df: pd.DataFrame,
        move_cols: list[str|int] | str | int | None = None,
        greater_or_equal: bool = True,
        add_inplace: bool = False) -> pd.DataFrame:
    """Return a DataFrame of 1s [True] and (0s) [False] in `move_cols`
    based on up and down moves along the rows of `df`.

    Args:
        df (pd.DataFrame): input DataFrame
        move_cols (list[str | int] | str | int | None, optional):
            columns to calculate up down moves on. Defaults to None.
        greater_or_equal (bool, optional): 
            if True, a >= b is 1 instead of a == b. 
            Defaults to True.
        add_inplace (bool, optional): if True, add the move columns inplace, 
            otherwise just return a DataFrame of `move_cols`.
            Defaults to False.

    Returns:
        pd.DataFrame: the DataFrame of moves
    """
    if add_inplace:
        _df = df
    else:
        _df = pd.DataFrame(index=df.index)
    if not isinstance(move_cols, list) and move_cols is not None:
        _move_cols = [move_cols]
    elif move_cols is None:
        _move_cols = df.columns.to_list()
    else:
        _move_cols = move_cols
    for move_col in resolve_col_names(df, _move_cols):
        vals = df[move_col].values
        if greater_or_equal:
            up = np.greater_equal(vals[1:], vals[:-1]).astype(float)
        else:
            up = np.greater(vals[1:], vals[:-1]).astype(float)
        _df.loc[_df.index[1:], move_col + MOVE_COL_EXT] = up
    return _df[1:]

def model_fit(
        model: torch.nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int,
        optimizer: Optimizer,
        batch_size: int = 8,
        shuffle_batches: bool = True,
        callbacks_make_fn: Callable | None = None,
        loss_criterion: torch.nn.modules.loss._Loss | Literal["mse"] = "mse",
        grad_clip: float | None = None,
        evaluate: bool = False) -> torch.nn.Module:
    """
    Train a PyTorch model.

    Args:
        model (torch.nn.Module): 
            The PyTorch model to be trained.

        X_train (torch.Tensor): 
            Input features for training.

        y_train (torch.Tensor): 
            Target values for training.

        epochs (int): 
            Number of epochs to train the model.

        optimizer (Optimizer): 
            Optimizer for model training.

        batch_size (int, optional): 
            Number of samples per batch. Defaults to 8.

        shuffle_batches (bool, optional): 
            If True, shuffles the data before creating batches.
                Defaults to True.

        callbacks_make_fn (Callable | None, optional): 
            A zero-argument callable that, if provided, is called after each \
                batch. If it returns True, training is interrupted. \
                    Defaults to None.

        loss_criterion 
            (torch.nn.modules.loss._Loss | Literal['mse'], optional): 
            Loss function to be minimized. Defaults to `mse`.

        grad_clip (float | None, optional):
            Maximum norm for gradient clipping. If None, no clipping is \
            applied. Defaults to None.

        evaluate (bool, optional): 
            If True, evaluates and prints the model’s error at \
                intervals during training. 
            Prints every `epochs // 5` epochs or at least \
                once per epoch if fewer than 5 epochs. 
            Defaults to False.

    Returns:
        torch.nn.Module: The trained model.

    Notes:
        - **Batch Handling**: If `batch_size` is greater than 1, \
            the model predicts for each batch.
        Otherwise, it performs prediction on the \
            last sample in each batch.
        - **Evaluation**: If `evaluate` is True, \
            error at intervals during training to track model progress.
    """
    if loss_criterion == 'mse':
        loss_criterion = nn.MSELoss()
    dataloader = data.DataLoader(
        data.TensorDataset(X_train, y_train), 
        shuffle=shuffle_batches,
        batch_size=batch_size)
    model.train()    
    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            if batch_size > 1:
                y_pred = model(X_batch)[:, -1, :]
            else:
                y_pred = model(X_batch)[-1, :]
            optimizer.zero_grad()
            loss = loss_criterion(y_pred, y_batch)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if callbacks_make_fn:
                if callbacks_make_fn(model) == True:
                    return model
        if evaluate and (epoch + 1) % max(1, epochs // 5) == 0:
            model.eval()
            with torch.no_grad():
                if len(X_train) > 1:
                    y_pred = model(X_train)[:, -1, :]
                else:
                    y_pred = model(X_train)[-1, :]
                train_err = loss_criterion(y_pred, y_train)
                print(f"Loss at epoch {epoch+1}: {train_err}")
            model.train()
    return model

def make_predictions(
        model: torch.nn.Module,
        first_test_data: torch.Tensor | np.ndarray,
        target_col_index: int = 0,
        n_forecasts: int = 1,
        scaler: sklearn.base.BaseEstimator | None = None,
        binary_classification: bool = False,
        binary_boundary: float = 0.5, # >=
        verbose: bool = False) -> tuple[torch.tensor, torch.tensor]:
    """
    Generate future predictions using a trained PyTorch model and returns
    a Tensor tuple of (all_cols_predictions, target_col_prediction),
    with optional inverse scaler transformation and binary classification.

    Args:
        model (torch.nn.Module): 
            The trained PyTorch model to use for making predictions.

        first_test_data (torch.Tensor | np.ndarray): 
            Initial test data to start the prediction sequence.

        target_col_index (int): 
            Index of the target column in the predictions for \
                extracting target-specific results.

        n_forecasts (int): 
            Number of future time steps to forecast.

        scaler (sklearn.base.BaseEstimator | None, optional): 
            Scaler used for inverse transforming the predictions, \
                if provided. Defaults to None.

        binary_classification (bool, optional): 
            If True, converts predictions to 1 if greater than or equal to \
                `binary_boundary` and 0 Otherwise. Defaults to False.

        binary_boundary (float, optional): 
            Threshold for binary classification. \
                Predictions greater than or equal to this value 
            are set to 1; otherwise, they are set to 0. \
                Only used if `binary_classification` is True. Defaults to 0.5.

        verbose (bool, optional): 
            If True, prints the progress of predictions. Defaults to False.

    Returns:
        tuple[torch.Tensor,torch.Tensor]: 
            - First element: All predictions as a tensor.
            - Second element: Predictions for the specified target column.
    """
    X_test = torch.stack([first_test_data])
    seq_len = X_test.shape[1]
    feature_count = X_test.shape[2]
    all_predictions = []
    with torch.no_grad():
        for i in range(n_forecasts):
            
            prediction = model(X_test.to(torch.float32))[
                :, seq_len-1:seq_len, :]
            if verbose:
                    print(f"Forecast [{i+1}/{n_forecasts}]: {prediction}")
            X_test = torch.cat((X_test, prediction), dim=1)[:, 1:, :]
            all_predictions.append(prediction)    
    all_pred_base = torch.stack(all_predictions).reshape(
        len(all_predictions), feature_count)
    if binary_classification:
        all_pred = torch.greater_equal(
            all_pred, binary_boundary).to(torch.float64)
    elif scaler is not None:
        try:
            all_pred = scaler.inverse_transform(all_pred_base.reshape(
                    n_forecasts, feature_count))
        except:
            all_pred = torch.tensor(
                    scaler.inverse_transform(all_pred.numpy().reshape(
                    n_forecasts, feature_count)))
    else:
        all_pred = all_pred_base
    return (all_pred, 
            all_pred.reshape(
                all_pred.shape[0], feature_count)[:, target_col_index])
    
# Returns a cleaned dataframe with no further filtration required
def _gen_data_df(
        df: pd.DataFrame, 
        future_moves_mode: bool, 
        target_col: str | int, 
        feature_cols: list[str|int]) -> pd.DataFrame:
    target_col_str_past = resolve_col_name(df, target_col)
    _feature_cols = _prep_feature_cols(df, feature_cols, target_col_str_past)
    if future_moves_mode:
        data_df = gen_moves(df, move_cols=_feature_cols)
    else:
        data_df = df.filter(_feature_cols)
    return fill_nan_numerics(data_df)

def _predict_the_future(
        df: pd.DataFrame,
        model: torch.nn.Module,
        optimizer: Optimizer | None = None,
        scaler: sklearn.base.BaseEstimator = MinMaxScaler(),
        prepped_df: pd.DataFrame | None = None,
        n_forecasts: int = 1,
        lookback: int = TRADING_MONTH,
        target_col: int | str = 0,
        feature_cols: list[str] | None = None,
        future_moves_mode: bool = False,
        move_cols: list[str] | None = None,
        sliding_window_divisor: int | None = None,
        epochs: int = 250,
        batch_size: int = 8,
        callbacks_make_fn: Callable | None = None, # model is the arg
        ignore_hols: list[str] = [],
        business_days_only: bool = True,
        return_only_target_cols: bool = True,
        verbose: bool = True) -> pd.DataFrame:
    if prepped_df is not None:
        data_df = prepped_df
    else:
        data_df = _gen_data_df(
            df, 
            future_moves_mode, 
            target_col, 
            feature_cols if move_cols is None else move_cols)
    scaled_data = torch.tensor(scaler.fit_transform(data_df))
    train_data, test_data = scaled_data[:-lookback], scaled_data[-lookback:]
    X, y = train_xy_split(train_data, lookback, sliding_window_divisor)
    model = model_fit(
        model, X, y, 
        epochs, optimizer, batch_size, 
        callbacks_make_fn=callbacks_make_fn, evaluate=verbose)
    target_col_str = resolve_col_name(data_df, target_col)
    target_col_index = data_df.columns.get_loc(target_col_str)
    fut_dates = make_future_date_list(
        df, n_forecasts, ignore_hols, business_days_only)
    prediction_df = pd.DataFrame(columns=df.columns, index=fut_dates)
    train_len = train_data.shape[0]
    test_data = scaled_data[train_len : train_len + lookback]
    all_predictions, _ = make_predictions(
        model, 
        test_data, 
        target_col_index, 
        n_forecasts=n_forecasts, 
        scaler=scaler, 
        binary_classification=future_moves_mode,
        verbose=verbose)
    for i in range(n_forecasts):
        for j in range(len(prediction_df.columns)):
            prediction_df.iloc[i, j] = all_predictions[i, j]
    if return_only_target_cols:
        # target_col_index:target_col_index+1 to ensure type == dataframe
        return (
            prediction_df.iloc[:, target_col_index : target_col_index + 1])
    else:
        return prediction_df

def predict_futures(
        df: pd.DataFrame,
        models: list[torch.nn.Module] | torch.nn.Module,
        optimizer: list[Optimizer] | Optimizer | None = None,
        scaler: sklearn.base.BaseEstimator = MinMaxScaler(),
        n_forecasts: int = 1,
        lookback: list[int] | np.ndarray | int = [TRADING_MONTH],
        future_moves_mode: bool = False,
        target_col: int | str = 0,
        feature_cols: list[str|None] | str | None = None,
        move_cols: list[str|None] | str | None = None,
        sliding_window_divisor: list[int] | np.ndarray | int | None = None,
        epochs: list[int] | np.ndarray | int = [250],
        batch_size: list[int] | np.ndarray | int = [8],
        callbacks_make_fn: Callable | None = None, # zero arg
        ignore_hols: list[list[str]] = [[]],
        return_only_target_cols: bool = True,
        verbose: bool = False) -> list[pd.DataFrame]:
    """
    Predict values using machine learning `models` on a \
    time series DataFrame. The returned DataFrame will be date-indexed,
    and might need to be re-indexed if `df` is not date-indexed.

    This function will do model training and, if `verbose`, evaluation \
    (eval at each fifth of total epochs). It enables time-series forecasting \
    with multiple models, allowing each model to handle \
    custom configurations of optimizer, feature columns, 
    lookback periods, and other training parameters. \
    It uses a custom broadcasting mechanism to automatically expand single \
    values into lists to match the number of models, facilitating flexible \
        configuration across multiple models.

    Args:
        df (pd.DataFrame): 
            Input DataFrame containing the time-series data for prediction.

        models (list[torch.nn.Module] | torch.nn.Module): 
            A single model or a list of PyTorch models used for prediction.

        optimizer (list[Optimizer] | Optimizer | None, optional): 
            Optimizer(s) for the model(s). If None, the default \
            optimizer (Adam) is used. Can be a single optimizer or a \
            list of optimizers corresponding to each model. Defaults to None.

        scaler (sklearn.base.BaseEstimator, optional): 
            Sklearn scaler used to normalize the data. \
            Defaults to MinMaxScaler().

        n_forecasts (int, optional): 
            The number of future time steps to predict. Defaults to 1.

        lookback (list[int] | np.ndarray | int, optional): 
            Lookback period(s) for each model, determining the number \
            of past time steps to consider. \
            Can be a single integer or a list/array of integers. 
            Defaults to `[TRADING_MONTH]`.     

        future_moves_mode (bool, optional): 
            Flag indicating whether to enable future moves mode, \
            which modifies the target prediction 
            to focus on directional moves (up or down) and appends "_UP" \
            to the target column name. Defaults to False.

        target_col (int | str, optional): 
            The column in `df` to be used as the target for prediction. \
            Can be specified by column index or column name. Defaults to 0.

        feature_cols (list[str | None] | str | None, optional): 
            Column(s) used as features for prediction. \
            Can be a single column name, a list of column names, \
            or None (using all columns as features). Defaults to None.

        move_cols (list[str | None] | str | None, optional): 
            Column(s) representing moves for future predictions, \
            similar to `feature_cols`. Defaults to None.

        sliding_window_divisor 
            (list[int] | np.ndarray | int | None, optional): 
            Factor(s) for sliding window adjustment, applicable per model. \
            During preparation of training data, select every `n` samples \
            where `n == lookback // sliding_window_divisor`.
            If None, no adjustment is applied, \
            and the window slides by 1 time step. Defaults to None.

        epochs (list[int] | np.ndarray | int, optional): 
            Number of training epochs for each model. \
            Can be a single integer or a list/array of integers. 
            Defaults to 250.

        batch_size (list[int] | np.ndarray | int, optional): 
            Batch size for training each model. \
            Can be a single integer or a list/array of integers. \
            Defaults to 8.

        callbacks_make_fn (Callable | None, optional): 
            A zero-argument callable that returns callbacks \
            for the training process, if needed. \
            If None, no callbacks are used. Defaults to None.

        ignore_hols (list[list[str]], optional): 
            List of holiday dates to ignore, per model. \
            Each element is a list of holiday dates as strings. \
            Defaults to `[[]]`.

        return_only_target_cols (bool, optional): 
            If True, only the target columns are returned in the output. \
            If False, all columns are returned. Defaults to True.

        verbose (bool, optional): 
            If True, provides detailed output during execution. \
            Defaults to False.

    Returns:
        list[pd.DataFrame]: 
            A list containing the prediction DataFrames for each model.

    Notes:
        - **Broadcasting Mechanism**: 
        This function uses a custom broadcasting approach to handle \
        parameters that can vary by model. \
        If a parameter is specified as a single value or a shorter list, \
        it is automatically expanded to match the number of models.
    """
    if not isinstance(models, list):
        models = [models]
    n_models = len(models)
    ret = []
    if isinstance(target_col, list):
        target_col = target_col[0]
    data_df = _gen_data_df(df, future_moves_mode, target_col, feature_cols)
    vect_ignore_hols = ignore_hols
    if isinstance(ignore_hols, list):
        if isinstance(ignore_hols[0], str):
            vect_ignore_hols = [vect_ignore_hols] * n_models
        elif isinstance(ignore_hols[0], list):
            vect_ignore_hols = _vect(ignore_hols, n_models)
    vect_optimizer = _vect(optimizer, n_models)
    for i in range(n_models):
        if vect_optimizer[i] is None:
            vect_optimizer[i] = adam(models[i])
    vect_feature_cols = _vect(feature_cols, n_models)
    vect_move_cols = _vect(move_cols, n_models)
    vect_lookback = _vect(lookback, n_models)
    vect_sliding_window_divisor = _vect(sliding_window_divisor, n_models)
    vect_epochs = _vect(epochs, n_models)
    vect_batch_size = _vect(batch_size, n_models)
    for i in range(n_models):
        if verbose:
            print(f"Running model {i}...")
        pred_df = _predict_the_future(
                df=df,
                model=models[i],
                optimizer=vect_optimizer[i],
                scaler=scaler,
                prepped_df=data_df,
                n_forecasts=n_forecasts,
                lookback=vect_lookback[i],
                target_col=target_col,
                feature_cols=vect_feature_cols[i],
                future_moves_mode=future_moves_mode,
                move_cols=vect_move_cols[i],
                sliding_window_divisor=vect_sliding_window_divisor[i],
                epochs=vect_epochs[i],
                batch_size=vect_batch_size[i],
                callbacks_make_fn=callbacks_make_fn,
                ignore_hols=vect_ignore_hols[i],
                return_only_target_cols=return_only_target_cols,
                verbose=verbose)
        ret.append(pred_df)
    return ret 

def historical_validation(
        df: pd.DataFrame,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: sklearn.base.BaseEstimator = MinMaxScaler(),
        df_steps_into_past: int = TRADING_MONTH * 3,
        target_col: int | str = 0,
        feature_cols: list[str] | None = None,
        future_moves_mode: bool = False,
        lookback: int = TRADING_MONTH * 2,
        forecast_len: int = 1,
        sliding_lookback_divisor: int | None = None,
        epochs: int = 250,
        return_only_target_col: bool = True,
        verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate a predictive model by generating predictions \
    against historical data. There will be `forecast_len` \
    time steps worth of predictions at each unskipped index, and \
    `forecast_len` number of indices in `df` will be skipped per forecast.
    Returns a (predictions, reality) tuple of DataFrames.

    Args:
        df (pd.DataFrame): Input DataFrame with historical time-series data.

        model (torch.nn.Module): The model to be validated.

        optimizer (torch.optim.Optimizer): Optimizer used for model training.

        scaler (sklearn.base.BaseEstimator, optional): 
            Scaler for data normalization. Defaults to MinMaxScaler().

        df_steps_into_past (int, optional): 
            Number of time steps to go back into past for validation. \
            Defaults to TRADING_MONTH * 3.

        target_col (int | str, optional): 
            Column index or name of the target variable. Defaults to 0.

        feature_cols (list[str] | None, optional): 
            List of feature columns to use. \
            If None, all columns are used. Defaults to None.

        future_moves_mode (bool, optional): 
            If True, generate up/down movement indicators. Defaults to False.

        lookback (int, optional): Lookback period for each input sequence. \
            Defaults to TRADING_MONTH * 2.

        forecast_len (int, optional): Number of time steps to predict \
            in each forecast step. Defaults to 1.

        sliding_window_divisor 
            (int | None, optional): 
            Factor for sliding window adjustment. \
            During preparation of training data, select every `n` samples \
            where `n == lookback // sliding_window_divisor`.
            If None, no adjustment is applied, \
            and the window slides by 1 time step. Defaults to None.

        epochs (int, optional): Number of training epochs. Defaults to 250.

        return_only_target_col (bool, optional): 
            If True, only return the target column in \
            predictions and validation. Defaults to True.

        verbose (bool, optional): 
            If True, print additional progress details. Defaults to False.

    Returns:
        tuple[pd.DataFrame,pd.DataFrame]: 
            - First element: DataFrame of model predictions.
            - Second element: DataFrame of actual historical \
            values for comparison.

    Raises:
        ValueError: If `df_steps_into_past` is less than `lookback`.

    Notes:
        - **Visualization**: To easily visualize the results of this function,
        you can call `quick_visuals(historical_validation(<<your args>>))`
    """
    if (df_steps_into_past < lookback):
        raise ValueError("df_steps_into_past must be >= lookback")
    target_col_str_past = resolve_col_name(df, target_col)
    _feature_cols = _prep_feature_cols(df, feature_cols, target_col_str_past)
    if future_moves_mode:
        data_df = gen_moves(df, move_cols=_feature_cols)
    else:
        data_df = df.filter(_feature_cols)
    data_df = fill_nan_numerics(data_df)
    scaled_data = torch.tensor(scaler.fit_transform(data_df))
    train_val_wall = lookback + df_steps_into_past
    val_df = data_df[-(train_val_wall):]
    train_data = scaled_data[:-(train_val_wall)]
    X, y = train_xy_split(train_data, lookback, sliding_lookback_divisor)
    model = model_fit(model, X, y, epochs, optimizer, evaluate=True)
    if not future_moves_mode:
        target_col_str_pred = target_col_str_past
    else:
        target_col_str_pred = resolve_col_name(
            df, target_col_str_past + MOVE_COL_EXT)
    target_col_index = data_df.columns.get_loc(target_col_str_pred)
    prediction_df = pd.DataFrame(
        columns=val_df.columns,
        index=val_df.index[-df_steps_into_past:])
    for i in range(0, df_steps_into_past, forecast_len):
        train_len = train_data.shape[0]
        test_data = scaled_data[
                        train_len + i : train_len + lookback + i]
        all_predictions, _ = make_predictions(
            model, 
            test_data, 
            target_col_index, 
            n_forecasts=forecast_len, 
            scaler=scaler,
            binary_classification=future_moves_mode,
            verbose=verbose)
        for j in range(len(prediction_df.columns)):
            for n in range(forecast_len):
                if i + n < df_steps_into_past:
                    prediction_df.iloc[i + n, j] = all_predictions[n, j]
    retloc_first = len(val_df) - df_steps_into_past
    retloc_second = len(val_df)
    if return_only_target_col:
        # target_col_index : target_col_index+1 is to ensure type == dataframe
        return (
            prediction_df.iloc[:, target_col_index : target_col_index + 1],
            val_df.iloc[retloc_first : retloc_second,
                            target_col_index : target_col_index + 1])
    else:
        return (prediction_df, val_df[retloc_first : retloc_second, :])

def quick_visuals(
        pred_list: (list[pd.DataFrame] | pd.DataFrame |
                    list[tuple] | tuple[pd.DataFrame, pd.DataFrame]),
        plot_title: str = "Forecast",
        add_lines: bool = True,
        linewidth: int = 3,
        markersize: int = 5,
        graphs_block_exec: bool = True):
    """Generate visualizations of forecasted data, \
        with optional comparison to actual past data.

    Args:
        pred_list (list[pd.DataFrame] | pd.DataFrame | list[tuple] \
                 | tuple[pd.DataFrame, pd.DataFrame]):
            A list of prediction DataFrames, \
            or a list of tuples where each tuple contains:
            -The first element as predicted values.
            -The second element (optional) as actual past \
values for comparison.
            Alternatively, a single DataFrame or \
tuple (prediction, past data) can be passed.

        plot_title (str, optional): 
            Title for the plot. Defaults to "Forecast".

        add_lines (bool, optional): 
            If True, lines connect the markers (solid lines). 
            If False, only markers are shown. Defaults to True.

        linewidth (int, optional): 
            Width of the lines connecting data points. Defaults to 3.

        markersize (int, optional): 
            Size of the markers for each data point. Defaults to 5.

        graphs_block_exec (bool, optional): 
            If True, blocks execution until the plot windows are closed. 
            Defaults to True.

    Notes:
        - **Multiple Forecasts**: Can handle multiple predictions, \
        plotting each entry in `pred_list` individually.
        If past data is provided, it is plotted alongside the \
            predictions for comparison.
        - **Legend**: A legend differentiates between \
            "Predictions" and "Reality" (if past data is included).
    """
    linestyle = "solid" if add_lines else "None"
    if not isinstance(pred_list, list):
        tup_list = [pred_list]
    else:
        tup_list = pred_list
    has_past = isinstance(tup_list[0], tuple)
    for i in range(len(tup_list)):
        plt.title(plot_title)
        pred = tup_list[i]
        past = tup_list[i]
        if has_past:
            pred = pred[0]
            past = past[1]
        plt.plot(
            pred, 
            label="Predictions", 
            marker="o", 
            markersize=markersize,
            linestyle=linestyle, 
            linewidth=linewidth)
        if has_past:
            plt.plot(
                past, 
                label="Reality",
                marker="x", 
                markersize=markersize,
                linestyle=linestyle, 
                linewidth=linewidth)
        plt.legend()
        plt.show()
    if graphs_block_exec:
        plt.close()

def aggregate_futures(
        pred_pasts: list[tuple[pd.DataFrame]] | list[pd.DataFrame], 
        method: str | Callable) -> pd.DataFrame:
    """Aggregate the results of a call to function predict_futures.
    The supported method strings are: 'mean', 'sum', 'min', 'max', 'count', 
    'std', 'var', 'median', 'prod', 'sem', 
    'skew', 'kurt', 'quantile', 'nunique', 'idxmin', 'idxmax', 
    'first', 'last', 'all' and 'any'.

    Args:
        pred_pasts (list[tuple[pd.DataFrame]] | list[pd.DataFrame]): 
            a list of (prediction DataFrame, past DataFrame) tuples or a 
            list of prediction DataFrame objects.
        method (str | Callable): 
            a string corresponding to the aggregation method to be used,
            or a function which takes a (concatenated) DataFrame to aggregate.
            E.g. 'mean' will call function pandas.DataFrame.mean.

    Raises:
        ValueError: if an unsupported method or method string is passed

    Returns:
        pd.DataFrame: result of the aggregation
    """
    tup_access = isinstance(pred_pasts[0], tuple)
    pred_list = []
    pred = None
    for i in range(len(pred_pasts)):
        pred = pred_pasts[i]
        if tup_access:
            pred = pred[0]
        pred_list.append(pred)
    if not isinstance(method, str) is not None:
        return method(pd.concat(pred_list, axis=1))
    elif method not in AGG_FUNC_DICT.keys():
        raise ValueError(f"Aggregate Method {method} not supported. "
                         f"These are:\n {AGG_FUNC_DICT.keys()}")
    else:
        return pd.DataFrame(
            AGG_FUNC_DICT[method](pd.concat(pred_list, axis=1), axis=1),
            columns=pred.columns)
    
def higher_final_pred(
        prediction: pd.DataFrame | list[pd.DataFrame],
        target_col: str | int = 0,
        equal_or_higher: bool = True) -> bool:
    """
    Determine if the final prediction value is higher than the initial value.

    Args:
        prediction (pd.DataFrame | list[pd.DataFrame]):
            The prediction data as a DataFrame or a list of DataFrames.

        target_col (str | int, optional): 
            Column to evaluate within the prediction DataFrame. \
                Can be specified by name or index.
            Defaults to the first column (0).

        equal_or_higher (bool, optional): 
            If True, checks if the final prediction is equal to or higher \
                than the initial prediction.
            If False, checks if the final prediction is strictly higher than \
                the initial prediction.
            Defaults to True.

    Returns:
        bool: True if the condition (equal or strictly higher) is met, \
            otherwise False.
    """
    preds = prediction.loc[:, resolve_col_name(prediction, target_col)]
    if equal_or_higher:
        return preds.iloc[-1] >= preds.iloc[0]
    else:
        return preds.iloc[-1] > preds.iloc[0]
    
def save_model(model: torch.nn.Module, filepath: str):
    """Save the trained model to a file.

    Args:
        model (torch.nn.Module): The trained model to save.
        filepath (str): The file path to save the model.
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(
        filepath: str, model_class: type, **model_kwargs) -> torch.nn.Module:
    """Load a model from a file in `eval` mode.

    Args:
        filepath (str): The file path to load the model from.
        model_class (type): The class of the model to load. 
            It should match the class used for training (e.g., `Magic`).
        **model_kwargs: Any additional arguments required for \
            the model's initialization.

    Returns:
        torch.nn.Module: The loaded model with weights restored.
    """
    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Model loaded from {filepath}")
    return model
