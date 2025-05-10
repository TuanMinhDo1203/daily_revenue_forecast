import pandas as pd

def load_data(path, file_type):
    """
    Load dữ liệu từ file CSV hoặc Excel.

    Parameters:
        path (str): Đường dẫn tới file dữ liệu.
        file_type (str): Loại file ('csv' hoặc 'excel').

    Returns:
        DataFrame: Dữ liệu đã đọc được từ file.
    """
    if file_type == "csv":
        return pd.read_csv(path)
    elif file_type == "excel":
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def drop_nan(df, subset=None):
    """
    Loại bỏ các dòng có giá trị NaN.

    Parameters:
        df (DataFrame): Dữ liệu đầu vào.
        subset (list or None): Cột cụ thể cần xét NaN (nếu có).

    Returns:
        DataFrame: Dữ liệu đã loại bỏ NaN.
    """
    return df.dropna(subset=subset)


def add_time_feature(df):
    """
    Thêm cột 'date' và đặc trưng ngày trong tuần từ cột 'Month' và 'NGÀY'.

    Parameters:
        df (DataFrame): Dữ liệu phải chứa cột 'Month' và 'NGÀY'.

    Returns:
        DataFrame: Dữ liệu đã thêm cột 'date' và one-hot weekday.
    """
    if 'Month' not in df.columns or 'NGÀY' not in df.columns:
        raise KeyError("Missing 'Month' or 'NGÀY' column in DataFrame.")
    
    df["date"] = pd.to_datetime({'year': 2023, 'month': df['Month'], 'day': df["NGÀY"]})
    df['weekday'] = df['date'].dt.day_name()
    df = pd.get_dummies(df, columns=['weekday'])
    return df


def add_external_features(df_main, df_external, on='date', drop_cols=None, how='inner'):
    """
    Gộp dữ liệu chính với dữ liệu bên ngoài (ví dụ: thời tiết).

    Parameters:
        df_main (DataFrame): Dữ liệu chính (ví dụ: doanh thu).
        df_external (DataFrame): Dữ liệu phụ (ví dụ: thời tiết).
        on (str): Tên cột dùng để merge (thường là 'date').
        drop_cols (list or None): Danh sách cột muốn drop sau khi merge.
        how (str): Kiểu merge ('inner', 'left', 'right', 'outer').

    Returns:
        DataFrame: Kết quả sau khi gộp và loại bỏ cột.
    """
    df_main[on] = pd.to_datetime(df_main[on])
    df_external[on] = pd.to_datetime(df_external[on])
    
    df_merged = pd.merge(df_main, df_external, on=on, how=how)
    
    if drop_cols:
        df_merged = df_merged.drop(columns=drop_cols)
    
    return df_merged


def add_time_series_features(df, revenue_col='revenue'):
    """
    Thêm các đặc trưng chuỗi thời gian từ cột doanh thu.

    Parameters:
        df (DataFrame): Dữ liệu phải chứa cột doanh thu.
        revenue_col (str): Tên cột doanh thu (mặc định là 'revenue').

    Returns:
        DataFrame: Dữ liệu với các đặc trưng chuỗi thời gian mới.
    """
    df = df.copy()

    df['revenue_prev1'] = df[revenue_col].shift(1)
    df['revenue_prev2'] = df[revenue_col].shift(2)

    df['revenue_mean_3'] = df[revenue_col].shift(1).rolling(window=3).mean()
    df['revenue_mean_7'] = df[revenue_col].shift(1).rolling(window=7).mean()

    df['revenue_diff'] = df['revenue_prev1'] - df['revenue_prev2']
    df['revenue_pct_change'] = (df['revenue_prev1'] - df['revenue_prev2']) / df['revenue_prev2']
    df['revenue_dev_from_mean7'] = df['revenue_prev1'] - df['revenue_mean_7']
    
    return df


def save_to_csv(df, path):
    """
    Lưu DataFrame thành file CSV.

    Parameters:
        df (DataFrame): Dữ liệu cần lưu.
        path (str): Đường dẫn tới file CSV muốn lưu.

    Returns:
        None
    """
    df.to_csv(path, index=False)
