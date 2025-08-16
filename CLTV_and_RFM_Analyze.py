################################################
# LIBRARIES
################################################
import pandas as pd
import numpy as np
from datetime import date, datetime
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

################################################
# SETTINGS
################################################
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)
pd.set_option("display.float_format", lambda x: "%.2f" % x)


################################################
# UTILITY FUNCTIONS
################################################

def comprehensive_date_detection(dataframe):
    """
       Detect and convert date columns in a pandas DataFrame.

       This function identifies date columns using three methods:
       1. Data type checking for datetime-like dtypes
       2. Column name matching against date-related keywords
       3. Content analysis with trial conversion for object-type columns

       Parameters
       ----------
       dataframe : pandas.DataFrame
           The input DataFrame to analyze for date columns.
           The DataFrame is modified in-place, with detected string date columns
           converted to datetime dtype.

       Returns
       -------
       list of str
           List of column names identified as date columns.

       Notes
       -----
       The function performs in-place modification of the input DataFrame:
       - String columns identified as dates are converted to datetime64[ns]
       - Invalid date values are converted to NaT (Not a Time)
       - Uses 'coerce' error handling for pd.to_datetime conversions

       The detection process uses the following criteria:
       - Columns already in datetime/date/time/period dtypes are automatically included
       - Columns with names containing date keywords are tested for conversion
       - Object-type columns are tested if they can't be converted to integers
       - A 95% success rate threshold is used for content-based detection

       Date keywords used for column name matching:
       'date', 'time', 'created', 'updated', 'modified', 'timestamp',
       'datetime', 'year', 'month', 'day'
       """
    date_cols = []

    # Keywords
    date_keywords = ['date', 'time', 'created', 'updated',
                     'modified', 'timestamp', 'datetime',
                     'year', 'month', 'day']

    for col in dataframe.columns:
        # 1. Dtype control
        if (dataframe[col].dtype.name.startswith(('datetime', 'date', 'time', 'period')) or
                str(dataframe[col].dtype) in ['datetime64[ns]', 'datetime64[ns, UTC]', 'timedelta64[ns]']):
            date_cols.append(col)
            print('The column', col, 'is a Date column because of its data type')
            continue

        # 2. Column Name Control
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in date_keywords):
            if dataframe[col].dtype == 'object':
                try:
                    test_conversion = pd.to_datetime(dataframe[col].dropna().head(10), errors='coerce')
                    if test_conversion.notna().any():
                        date_cols.append(col)
                        dataframe[col] = pd.to_datetime(dataframe[col], errors='coerce')
                        print('The column', col, 'is a Date column because of its column name')
                        continue
                except:
                    pass

        # 3. Value Control for String date
        try:
            dataframe[col].astype(int)
            might_date = False
        except:
            might_date = True

        if dataframe[col].dtype == 'object' and col not in date_cols and might_date:
            try:
                sample = dataframe[col].dropna().head(50)
                if len(sample) > 5:
                    # pd.to_datetime test
                    converted = pd.to_datetime(sample, errors='coerce')
                    success_rate = converted.notna().sum() / len(sample)

                    if success_rate >= 0.95:
                        date_cols.append(col)
                        print('The column', col, 'is a Date columns because of the data type change trial with success rate:', success_rate)
                        dataframe[col] = pd.to_datetime(dataframe[col], errors='coerce')
            except:
                pass

    return date_cols


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Classify DataFrame columns into categorical, numerical, cardinal, and date categories.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame whose columns will be classified.
    cat_th : int, optional
        Threshold for determining categorical columns from numerical data (default is 10).
    car_th : int, optional
        Threshold for determining cardinal columns from categorical data (default is 20).

    Returns
    -------
    cat_cols : list
        List of categorical column names.
    num_cols : list
        List of numerical column names.
    cat_but_car : list
        List of categorical but cardinal column names.
    date_cols : list
        List of date/datetime column names.
    """

    # First find and change the date cols
    comprehensive_date_detection(dataframe)

    # Identify date/datetime columns
    date_cols = [col for col in dataframe.columns if
                 dataframe[col].dtype.name.startswith(('datetime', 'date', 'time', 'period')) or
                 str(dataframe[col].dtype) in ['datetime64[ns]', 'datetime64[ns, UTC]', 'timedelta64[ns]']]

    # Identify base categorical columns
    cat_cols = [col for col in dataframe.columns if
                str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    # Find numerical columns that behave like categorical
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int", "float"]]

    # Identify high cardinality categorical columns
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]
    cat_but_car = [col for col in cat_but_car if col not in date_cols]

    # Combine categorical columns
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Identify numerical columns
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    # Print summary
    print(f"Total Observations: {dataframe.shape[0]}")
    print(f"Total Variables: {dataframe.shape[1]}")
    print(f'🏷️  Categorical columns: {len(cat_cols)} ({len(cat_cols) / dataframe.shape[1] * 100:5.1f}%)')
    print(f'🔢  Numerical columns: {len(num_cols)} ({len(num_cols) / dataframe.shape[1] * 100:5.1f}%)')
    print(f'🎯  Cardinal columns: {len(cat_but_car)} ({len(cat_but_car) / dataframe.shape[1] * 100:5.1f}%)')
    print(f'🔄  Numerical but categorical: {len(num_but_cat)} ({len(num_but_cat) / dataframe.shape[1] * 100:5.1f}%)')
    print(f'📅  Date columns: {len(date_cols)} ({len(date_cols) / dataframe.shape[1] * 100:5.1f}%)')

    return cat_cols, num_cols, cat_but_car, date_cols


def outlier_threshold_IQR(dataframe, variable, q1=0.25, q3=0.75, show_plt=False):
    """
    Calculate outlier thresholds using IQR method.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame containing the variable.
    variable : str
        Name of the numerical column to analyze.
    q1 : float, optional
        Lower quantile for IQR calculation (default is 0.25).
    q3 : float, optional
        Upper quantile for IQR calculation (default is 0.75).
    show_plt : bool, optional
        Whether to show boxplot visualization (default is False).

    Returns
    -------
    tuple
        Lower and upper outlier thresholds.
    """
    quantile_1 = dataframe[variable].quantile(q1)
    quantile_3 = dataframe[variable].quantile(q3)
    interquantile_range = quantile_3 - quantile_1
    upper_limit = quantile_3 + 1.5 * interquantile_range
    lower_limit = quantile_1 - 1.5 * interquantile_range

    if show_plt:
        plt.figure(figsize=(10, 6))
        plt.boxplot(dataframe[variable], patch_artist=True)
        plt.axhline(y=upper_limit, color='red', linestyle='--', linewidth=2,
                    label=f'Upper Limit: {upper_limit:.2f}')
        plt.axhline(y=lower_limit, color='red', linestyle='--', linewidth=2,
                    label=f'Lower Limit: {lower_limit:.2f}')
        plt.title(f'IQR Boxplot for {variable}')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show(block=True)

    return round(lower_limit), round(upper_limit)


def replace_with_threshold(dataframe, variable, q1=0.25, q3=0.75, replace_lower=False, show_plt=False):
    """
    Replace outliers with threshold values using IQR method.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame containing the variable (modified in-place).
    variable : str
        Name of the numerical column to process.
    q1 : float, optional
        Lower quantile for IQR calculation (default is 0.25).
    q3 : float, optional
        Upper quantile for IQR calculation (default is 0.75).
    replace_lower : bool, optional
        Whether to replace lower outliers as well (default is False).
    """
    lower_limit, upper_limit = outlier_threshold_IQR(dataframe, variable, q1, q3)

    if replace_lower:
        dataframe.loc[(dataframe[variable] < lower_limit), variable] = lower_limit

    dataframe.loc[(dataframe[variable] > upper_limit), variable] = upper_limit

    if show_plt:
        plt.figure(figsize=(10, 6))
        plt.boxplot(dataframe[variable], patch_artist=True)
        plt.axhline(y=upper_limit, color='red', linestyle='--', linewidth=2,
                    label=f'Upper Limit: {upper_limit:.2f}')
        plt.axhline(y=lower_limit, color='red', linestyle='--', linewidth=2,
                    label=f'Lower Limit: {lower_limit:.2f}')
        plt.title(f'Replaced Threshold Boxplot for {variable}')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show(block=True)


def check_df_tabulate(dataframe, head=10, detailed_stats=True, show_correlations=True,
                      plot=False, cat_th=10, car_th=20, q1=0.25, q3=0.75):
    """
    Generate a comprehensive report of a pandas DataFrame with formatted tables and visualizations.

    This function provides a detailed analysis of a DataFrame including basic information,
    column statistics, numerical summaries, categorical analysis, outlier detection,
    correlation analysis, and optional visualizations using tabulate for enhanced formatting.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame to analyze and report on.
    head : int, optional
        Number of rows to display for head and tail sections (default is 10).
    detailed_stats : bool, optional
        Whether to include detailed statistical analysis (skewness, kurtosis, outliers) (default is True).
    show_correlations : bool, optional
        Whether to show correlation analysis for numerical columns (default is True).
    plot : bool, optional
        Whether to show visualizations (histograms, boxplots, heatmaps) (default is False).
    cat_th : int, optional
        Threshold for categorical classification in grab_col_names (default is 10).
    car_th : int, optional
        Threshold for cardinal classification in grab_col_names (default is 20).
    q1 : float, optional
        Lower quantile for outlier analysis (default is 0.25).
    q3 : float, optional
        Upper quantile for outlier analysis (default is 0.75).

    Returns
    -------
    None
        This function prints the report directly to console and does not return any value.

    Notes
    -----
    The function requires the 'tabulate' library to be installed for proper formatting.
    For visualizations, 'matplotlib' and 'seaborn' libraries are required.
    The report includes:
    - Basic DataFrame information (shape, memory usage)
    - Column classification using grab_col_names function
    - Column details (data types, unique values, missing values)
    - Statistical summaries for numerical columns
    - Distribution shape analysis (skewness, kurtosis) with optional histograms
    - Categorical analysis (top values, frequency distribution)
    - Outlier analysis using outlier_threshold function with optional boxplots
    - Correlation analysis for numerical variables with optional heatmap
    - Sample data from beginning and end of DataFrame
    """


    try:
        import pandas as pd
        import numpy as np
        from tabulate import tabulate
    except ImportError:
        print("⚠️ Warning: pands/numpy/tabulate not found. Function will not work. Try to install libs yourself.")

    # Import libraries for plotting if needed
    if plot:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.style.use('default')
        except ImportError:
            print("⚠️ Warning: matplotlib/seaborn not found. Plots will be skipped.")
            plot = False

    print("#" * 84)
    print("#" * 27, " " * 5, "DATAFRAME REPORT", " " * 5, "#" * 27)
    print("#" * 84)

    # 0. Sütun Sınıflandırması / Column Classification using grab_col_names
    print("\n🏷️ COLUMN CLASSIFICATION / SÜTUN SINIFLANDIRMASI")
    cat_cols, num_cols, cat_but_car, date_cols = grab_col_names(dataframe, cat_th, car_th)

    # 1. Temel Bilgiler / Basic Information
    basic_data = [
        ["Number of Rows / Satır Sayısı", dataframe.shape[0]],
        ["Number of Columns / Sütun Sayısı", dataframe.shape[1]],
        ["Total Number of Cells / Toplam Hücre Sayısı", dataframe.shape[0] * dataframe.shape[1]],
        ["Memory (MB) / Bellek (MB)", round(dataframe.memory_usage(deep=True).sum() / 1024 ** 2, 2)],
        ["Numerical Columns / Sayısal Sütun", len(num_cols)],
        ["Categorical Columns / Kategorik Sütun", len(cat_cols)],
        ["Cardinal Columns / Kardinal Sütun", len(cat_but_car)],
        ["Date Columns / Tarih Sütunu", len(date_cols)]
    ]

    print("\n\n\n📊 BASIC INFORMATION / TEMEL BİLGİLER")
    print(tabulate(basic_data, headers=["Metric / Metrik", "Value / Değer"], tablefmt="fancy_grid"))

    # 2. Sütun Bilgileri / Column Information
    column_data = []
    for col in dataframe.columns:
        # Sütun tipini belirle
        if col in num_cols:
            col_type = "Numerical"
        elif col in cat_cols:
            col_type = "Categorical"
        elif col in cat_but_car:
            col_type = "Cardinal"
        elif col in date_cols:
            col_type = "Date"
        else:
            col_type = "Other"

        column_data.append([
            col,
            str(dataframe[col].dtype),
            col_type,
            dataframe[col].nunique(),
            dataframe[col].isnull().sum(),
            f"{round((dataframe[col].isnull().sum() / len(dataframe)) * 100, 2)}%"
        ])

    print("\n\n\n📋 COLUMN INFORMATION / SÜTUN BİLGİLERİ")
    print(tabulate(column_data,
                   headers=["Column / Sütun", "Data Type / Veri Tipi", "Classification / Sınıf",
                            "Unique Values / Benzersiz Değerler", "NaN Count / NaN Sayısı", "NaN % / NaN Yüzdesi"],
                   tablefmt="fancy_grid"))

    # 3. İstatistikler (Sayısal sütunlar için) / Statistics (for Numerical columns)
    if len(num_cols) > 0:
        print("\n\n\n📈 STATISTICS OF NUMERICAL COLUMNS / SAYISAL SÜTUNLARIN İSTATİSTİKLERİ")
        stats_data = []
        for col in num_cols:
            col_data = dataframe[col].dropna()  # NaN değerleri çıkar
            if len(col_data) > 0:
                # Mod hesaplama (güvenli)/ Mode calculation (safe way)
                mode_val = col_data.mode()
                mode_display = round(mode_val.iloc[0], 2) if len(mode_val) > 0 else "N/A"

                stats_data.append([
                    col,
                    round(col_data.mean(), 2),
                    mode_display,
                    round(col_data.std(), 2),
                    col_data.min(),
                    round(col_data.quantile(0.01), 2),
                    round(col_data.quantile(0.1), 2),
                    round(col_data.quantile(0.25), 2),
                    round(col_data.median(), 2),
                    round(col_data.quantile(0.75), 2),
                    round(col_data.quantile(0.9), 2),
                    round(col_data.quantile(0.99), 2),
                    col_data.max()
                ])

        print(tabulate(stats_data,
                       headers=["Column / Sütun", "Mean / Ort", "Mode / Mod", "Std / Std Sap", "Min / Min", "%1", "%10",
                                "Q1", "Median / Medyan (Q2)", "Q3", "%90", "%99", "Max / Maks"],
                       tablefmt="fancy_outline"))

    # 4. Dağılım Şekli Analizi / Distribution Shape Analysis
    if detailed_stats and len(num_cols) > 0:
        print("\n\n\n📐 DISTRIBUTION SHAPE ANALYSIS / DAĞILIM ŞEKLİ ANALİZİ")
        shape_data = []
        for col in num_cols:
            col_data = dataframe[col].dropna()
            if len(col_data) > 0:
                skew_val = col_data.skew()
                kurt_val = col_data.kurtosis()

                # Çarpıklık yorumu
                if abs(skew_val) < 0.5:
                    skew_interp = "Symmetric / Simetrik"
                    skew_symbol = "⚖️"
                elif skew_val > 0.5:
                    skew_interp = "Right Skewed / Sağa Çarpık"
                    skew_symbol = "↗️"
                else:
                    skew_interp = "Left Skewed / Sola Çarpık"
                    skew_symbol = "↖️"

                # Basıklık yorumu
                if kurt_val > 0:
                    kurt_interp = "Peaked / Sivri"
                    kurt_symbol = "🔺"
                elif kurt_val < 0:
                    kurt_interp = "Flat / Basık"
                    kurt_symbol = "⬜"
                else:
                    kurt_interp = "Normal"
                    kurt_symbol = "🔵"

                shape_data.append([
                    col,
                    round(skew_val, 3),
                    f"{skew_symbol} {skew_interp}",
                    round(kurt_val, 3),
                    f"{kurt_symbol} {kurt_interp}"
                ])

        print(tabulate(shape_data,
                       headers=["Column / Sütun", "Skewness / Çarpıklık", "Skew Interpretation / Çarpıklık Yorumu",
                                "Kurtosis / Basıklık", "Kurt Interpretation / Basıklık Yorumu"],
                       tablefmt="fancy_outline"))

        # Distribution Plots / Dağılım Grafikleri
        if plot and len(num_cols) > 0:
            print("\n📊 DISTRIBUTION PLOTS / DAĞILIM GRAFİKLERİ")
            n_cols = min(3, len(num_cols))
            n_rows = (len(num_cols) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()

            for idx, col in enumerate(num_cols):
                col_data = dataframe[col].dropna()
                ax = axes[idx] if len(num_cols) > 1 else axes[0]

                # Histogram + KDE
                ax.hist(col_data, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
                try:
                    col_data.plot.kde(ax=ax, color='red', linewidth=2)
                except:
                    pass

                ax.set_title(f'{col}\nSkew: {col_data.skew():.3f}, Kurt: {col_data.kurtosis():.3f}')
                ax.set_xlabel('Values')
                ax.set_ylabel('Density')
                ax.grid(True, alpha=0.3)

            # Boş subplot'ları gizle
            for idx in range(len(num_cols), len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()
            plt.show(block=True)

    # 5. Kategorik Sütun Analizi / Categorical Analysis
    if len(cat_cols) > 0:
        print("\n\n\n📊 CATEGORICAL COLUMNS ANALYSIS / KATEGORİK SÜTUN ANALİZİ")
        for col in cat_cols:
            col_data = dataframe[col].dropna()
            if len(col_data) > 0:
                print(f"\n🏷️ {col.upper()} - Top 10 Values / En Sık 10 Değer:")
                value_counts = col_data.value_counts().head(10)
                cat_analysis_data = []
                for value, count in value_counts.items():
                    percentage = (count / len(col_data)) * 100
                    cat_analysis_data.append([value, count, f"{percentage:.2f}%"])

                print(tabulate(cat_analysis_data,
                               headers=["Value / Değer", "Count / Sayı", "Percentage / Yüzde"],
                               tablefmt="rounded_outline"))

    # 6. Aykırı Değer Analizi / Outlier Analysis
    if detailed_stats and len(num_cols) > 0:
        print(f"\n\n\n⚠️ OUTLIER ANALYSIS (Q1={q1}, Q3={q3}) / AYKIRI DEĞER ANALİZİ")
        outlier_data = []
        for col in num_cols:
            col_data = dataframe[col].dropna()
            if len(col_data) > 0:
                try:
                    lower_limit, upper_limit = outlier_threshold_IQR(dataframe, col, q1, q3)

                    # Aykırı değer sayısını hesapla
                    outliers_lower = len(col_data[col_data < lower_limit])
                    outliers_upper = len(col_data[col_data > upper_limit])
                    total_outliers = outliers_lower + outliers_upper
                    outlier_percentage = (total_outliers / len(col_data)) * 100

                    outlier_data.append([
                        col,
                        f"Q1({q1})-1.5 IQR",
                        f"Q3({q3})+1.5*IQR",
                        lower_limit,
                        upper_limit,
                        outliers_lower,
                        outliers_upper,
                        total_outliers,
                        f"{outlier_percentage:.2f}%"
                    ])
                except:
                    # Eğer outlier_threshold fonksiyonu hata verirse
                    outlier_data.append([col, "Error", "Error", "Error", "Error", "Error", "Error", "Error", "Error"])

        print(tabulate(outlier_data,
                       headers=["Column / Sütun", "Lower Method / Alt Yöntem", "Upper Method / Üst Yöntem",
                                "Lower Limit / Alt Sınır", "Upper Limit / Üst Sınır",
                                "Lower Outliers / Alt Aykırı", "Upper Outliers / Üst Aykırı",
                                "Total Outliers / Toplam Aykırı", "Outlier % / Aykırı %"],
                       tablefmt="fancy_outline"))

        # Outlier Boxplots / Aykırı Değer Kutu Grafikleri
        if plot and len(num_cols) > 0:
            print("\n📦 OUTLIER BOXPLOTS / AYKIRI DEĞER KUTU GRAFİKLERİ")
            n_cols = min(3, len(num_cols))
            n_rows = (len(num_cols) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()

            for idx, col in enumerate(num_cols):
                ax = axes[idx] if len(num_cols) > 1 else axes[0]

                # Boxplot
                box_data = dataframe[col].dropna()
                ax.boxplot(box_data, patch_artist=True)

                # Outlier limits çizgileri
                try:
                    lower_limit, upper_limit = outlier_threshold_IQR(dataframe, col, q1, q3)
                    ax.axhline(y=upper_limit, color='red', linestyle='--', alpha=0.7,
                               label=f'Upper Limit: {upper_limit}')
                    ax.axhline(y=lower_limit, color='red', linestyle='--', alpha=0.7,
                               label=f'Lower Limit: {lower_limit}')
                    ax.legend()
                except:
                    pass

                ax.set_title(f'{col} - Outlier Detection')
                ax.set_ylabel('Values')
                ax.grid(True, alpha=0.3)

            # Boş subplot'ları gizle
            for idx in range(len(num_cols), len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()
            plt.show(block=True)

    # 7. Korelasyon Analizi / Correlation Analysis
    if show_correlations and len(num_cols) > 1:
        print("\n\n\n🔗 CORRELATION ANALYSIS / KORELASYON ANALİZİ")

        corr_matrix = dataframe[num_cols].corr()

        # Eliminating repaet
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                if not pd.isna(corr_val):
                    corr_pairs.append([f"{col1} - {col2}", round(corr_val, 3)])

        # Ordering according to Carr value
        corr_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        print("🔝 Top 10 Correlations / En Yüksek 10 Korelasyon:")
        print(tabulate(corr_pairs[:10],
                       headers=["Variable Pair / Değişken Çifti", "Correlation / Korelasyon"],
                       tablefmt="rounded_outline"))

        # Korelasyon Matrisi Tablosu (eğer sütun sayısı çok fazla değilse)
        if len(num_cols) <= 10:
            print(f"\n📋 CORRELATION MATRIX / KORELASYON MATRİSİ")
            # Korelasyon matrisini tabulate için hazırla
            corr_table_data = []
            for i, row_name in enumerate(corr_matrix.index):
                row_data = [row_name]
                for j, col_name in enumerate(corr_matrix.columns):
                    val = corr_matrix.iloc[i, j]
                    if pd.isna(val):
                        row_data.append("-")
                    elif i == j:
                        row_data.append("1.00")
                    else:
                        row_data.append(f"{val:.3f}")
                corr_table_data.append(row_data)

            headers = ["Variables"] + list(corr_matrix.columns)
            print(tabulate(corr_table_data, headers=headers, tablefmt="fancy_grid"))
        else:
            print(
                f"\n⚠️ Too many numerical columns ({len(num_cols)}) for matrix display. Showing top correlations only.")

        # Correlation Heatmap / Korelasyon Isı Haritası
        if plot and len(num_cols) > 1:
            print("\n🔥 CORRELATION HEATMAP / KORELASYON ISI HARİTASI")
            plt.figure(figsize=(10, 8))

            # Heatmap oluştur
            mask = None
            if len(num_cols) > 2:
                # Üst üçgeni maskelemek için (daha temiz görünüm)
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

            sns.heatmap(corr_matrix,
                        annot=True,
                        cmap='RdBu_r',
                        center=0,
                        fmt='.3f',
                        square=True,
                        mask=mask,
                        cbar_kws={"shrink": .8})

            plt.title('Correlation Heatmap / Korelasyon Isı Haritası')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show(block=True)

            print()

    # 8. Cardinal Sütunlar (Yüksek Kardinaliteli) / Cardinal Columns
    if len(cat_but_car) > 0:
        print("\n\n\n🎯 CARDINAL (HIGH CARDINALITY) COLUMNS / KARDINAL SÜTUNLAR")
        cardinal_data = []
        for col in cat_but_car:
            unique_count = dataframe[col].nunique()
            null_count = dataframe[col].isnull().sum()
            cardinal_data.append([
                col,
                unique_count,
                null_count,
                f"Too many unique values / Çok fazla benzersiz değer ({unique_count})"
            ])

        print(tabulate(cardinal_data,
                       headers=["Column / Sütun", "Unique Count / Benzersiz Sayı",
                                "NaN Count / NaN Sayısı", "Note / Not"],
                       tablefmt="fancy_grid"))

    # 9. İlk satırlar / Head
    print(f"\n\n\n🔝 FIRST {head} ROWS / İLK {head} SATIR")
    print(tabulate(dataframe.head(head), headers='keys', tablefmt="rounded_outline"))

    # 10. Son satırlar / Tail
    print(f"\n🔚 LAST {head} ROWS / SON {head} SATIR")
    print(tabulate(dataframe.tail(head), headers='keys', tablefmt="rounded_outline"))

    print("\n\n\n" + "#" * 84)
    print("#" * 33, "REPORT COMPLETED", "#" * 33)
    print("#" * 84)



################################################
# DATA LOADING AND PREPROCESSING
################################################

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the online retail data.

    Parameters
    ----------
    file_path : str
        Path to the Excel file.

    Returns
    -------
    pandas.DataFrame
        Preprocessed DataFrame.
    """
    # Load data from both sheets
    df_2009_2010 = pd.read_excel(file_path, sheet_name="Year 2009-2010")
    df_2010_2011 = pd.read_excel(file_path, sheet_name="Year 2010-2011")

    # Combine datasets
    df = pd.concat([df_2009_2010, df_2010_2011], axis=0, ignore_index=True)

    # Basic preprocessing
    df.dropna(inplace=True)  # Remove missing values
    df = df[~df["Invoice"].str.contains("C", na=False)]  # Remove cancelled orders
    df["TotalPrice"] = df["Quantity"] * df["Price"]  # Calculate total price

    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


################################################
# RFM ANALYSIS
################################################

def create_rfm_table(dataframe, customer_id_col="Customer ID",
                     invoice_date_col="InvoiceDate", invoice_col="Invoice",
                     total_price_col="TotalPrice", later_day_analyze=2):
    """
    Create RFM (Recency, Frequency, Monetary) table.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The input DataFrame.
    customer_id_col : str
        Name of the customer ID column.
    invoice_date_col : str
        Name of the invoice date column.
    invoice_col : str
        Name of the invoice column.
    total_price_col : str
        Name of the total price column.
    later_day_analyze : int
        Number of days for adding to max date in the dataframe so that analyze date determined.

    Returns
    -------
    pandas.DataFrame
        RFM table with scores and segments.
    """

    # Calculate analysis date (2 days after the latest transaction)
    analyze_date = dataframe[invoice_date_col].max() + pd.Timedelta(days=later_day_analyze)

    # Calculate RFM metrics
    rfm = dataframe.groupby(customer_id_col).agg({
        invoice_date_col: lambda x: (analyze_date - x.max()).days,  # Recency
        invoice_col: lambda x: x.nunique(),  # Frequency
        total_price_col: lambda x: x.sum()  # Monetary
    }).reset_index()

    rfm.columns = [customer_id_col, "Recency", "Frequency", "Monetary"]

    # Filter positive monetary values
    rfm = rfm[rfm["Monetary"] > 0]

    # Calculate RFM scores
    rfm["Recency_Score"] = pd.qcut(rfm['Recency'].rank(method="first"), 5, labels=[5, 4, 3, 2, 1])
    rfm["Frequency_Score"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["Monetary_Score"] = pd.qcut(rfm['Monetary'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

    # Create RFM Score
    rfm["RFM_Score"] = (rfm["Recency_Score"].astype(str) + rfm["Frequency_Score"].astype(str))

    # Define segments
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['Segment'] = rfm['RFM_Score'].replace(seg_map, regex=True)
    rfm["RFM_Score"] = rfm["RFM_Score"].astype(int)

    print("\n\n\n" + "#" * 84)
    print("#" * 30, " " * 5, "RFM  TABLE", " " * 5, "#" * 30)
    print("#" * 84)
    print(rfm)

    return rfm



def analyze_rfm_segments(rfm_df, sort_by="segment_hierarchy"):
    """
    Analyze RFM segments and provide insights.

    Parameters
    ----------
    rfm_df : pandas.DataFrame
        RFM DataFrame with segments.
    sort_by : str, optional
        Sorting method for segments:
        - "segment_hierarchy": Sort by business value hierarchy (default)
        - "customer_count": Sort by number of customers (descending)
        - "monetary": Sort by average monetary value (descending)
        - "frequency": Sort by average frequency (descending)
        - "recency": Sort by average recency (ascending - better recency first)
        - "alphabetical": Sort alphabetically by segment name
    """

    print("\n\n\n" + "#" * 84)
    print("#" * 25, " " * 5, "RFM SEGMENT ANALYSIS", " " * 5, "#" * 25)
    print("#" * 84)

    # Calculate segment analysis
    segment_analysis = rfm_df.groupby('Segment').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']
    }).round(2)

    # Define segment hierarchy (from best to worst in business value)
    segment_hierarchy = {
        'champions': 1,
        'loyal_customers': 2,
        'potential_loyalists': 3,
        'new_customers': 4,
        'promising': 5,
        'need_attention': 6,
        'about_to_sleep': 7,
        'at_risk': 8,
        'cant_loose': 9,
        'hibernating': 10
    }

    # Prepare data for tabulate with sorting
    segment_data = []
    for segment in segment_analysis.index:
        recency_mean = segment_analysis.loc[segment, ('Recency', 'mean')]
        frequency_mean = segment_analysis.loc[segment, ('Frequency', 'mean')]
        monetary_mean = segment_analysis.loc[segment, ('Monetary', 'mean')]
        customer_count = int(segment_analysis.loc[segment, ('Monetary', 'count')])

        segment_data.append([
            segment,
            f"{recency_mean:.1f}",
            f"{frequency_mean:.1f}",
            f"{monetary_mean:.2f}",
            customer_count,
            recency_mean,  # For sorting purposes
            frequency_mean,  # For sorting purposes
            monetary_mean,  # For sorting purposes
            segment_hierarchy.get(segment, 99)  # For hierarchy sorting
        ])

    # Sort the data based on the specified method
    if sort_by == "segment_hierarchy":
        segment_data.sort(key=lambda x: x[8])  # Sort by hierarchy value
        sort_description = "Business Value Hierarchy (Best to Worst)"
    elif sort_by == "customer_count":
        segment_data.sort(key=lambda x: x[4], reverse=True)  # Sort by customer count (descending)
        sort_description = "Customer Count (Highest to Lowest)"
    elif sort_by == "monetary":
        segment_data.sort(key=lambda x: x[7], reverse=True)  # Sort by monetary (descending)
        sort_description = "Average Monetary Value (Highest to Lowest)"
    elif sort_by == "frequency":
        segment_data.sort(key=lambda x: x[6], reverse=True)  # Sort by frequency (descending)
        sort_description = "Average Frequency (Highest to Lowest)"
    elif sort_by == "recency":
        segment_data.sort(key=lambda x: x[5])  # Sort by recency (ascending - lower is better)
        sort_description = "Average Recency (Most Recent First)"
    else:  # alphabetical
        segment_data.sort(key=lambda x: x[0])  # Sort by segment name
        sort_description = "Alphabetical Order"

    # Remove the extra columns used for sorting
    display_data = [[row[0], row[1], row[2], row[3], row[4]] for row in segment_data]

    print(f"📊 RFM SEGMENT SUMMARY (Sorted by: {sort_description}):")
    print(tabulate(display_data,
                   headers=["Segment", "Avg Recency (days)", "Avg Frequency", "Avg Monetary ($)", "Customer Count"],
                   tablefmt="fancy_grid"))

    # Segment distribution with percentages (also sorted the same way)
    segment_counts = rfm_df['Segment'].value_counts()
    segment_percentages = (segment_counts / len(rfm_df) * 100).round(2)

    distribution_data = []
    for row in segment_data:
        segment = row[0]
        distribution_data.append([
            segment,
            segment_counts[segment],
            f"{segment_percentages[segment]:.2f}%"
        ])

    print(f"\n📈 SEGMENT DISTRIBUTION (Sorted by: {sort_description}):")
    print(tabulate(distribution_data,
                   headers=["Segment", "Customer Count", "Percentage"],
                   tablefmt="fancy_outline"))

    # Enhanced insights with ranking information
    print(f"\n💡 KEY INSIGHTS:")
    insights_data = []

    # Top segments by customer count
    top_segment_by_count = max(segment_data, key=lambda x: x[4])
    insights_data.append(["Largest Segment", top_segment_by_count[0],
                          f"{top_segment_by_count[4]} customers ({segment_percentages[top_segment_by_count[0]]:.1f}%)"])

    # Highest value segment
    highest_value_segment = max(segment_data, key=lambda x: x[7])
    insights_data.append(
        ["Highest Value Segment", highest_value_segment[0], f"${highest_value_segment[7]:.2f} avg monetary"])

    # Most frequent segment
    most_frequent_segment = max(segment_data, key=lambda x: x[6])
    insights_data.append(
        ["Most Frequent Segment", most_frequent_segment[0], f"{most_frequent_segment[6]:.1f} avg frequency"])

    # Most recent segment (lowest recency)
    most_recent_segment = min(segment_data, key=lambda x: x[5])
    insights_data.append(
        ["Most Recent Segment", most_recent_segment[0], f"{most_recent_segment[5]:.1f} days avg recency"])

    # Business priority segments
    priority_segments = [row[0] for row in segment_data[:3]]  # Top 3 based on current sorting
    insights_data.append(
        ["Top 3 Priority Segments", ", ".join(priority_segments), f"Based on {sort_description.lower()}"])

    print(tabulate(insights_data,
                   headers=["Metric", "Segment", "Value"],
                   tablefmt="rounded_outline"))

    # Add segment hierarchy explanation if using hierarchy sorting
    if sort_by == "segment_hierarchy":
        print(f"\n🏆 SEGMENT HIERARCHY EXPLANATION:")
        hierarchy_explanation = [
            ["1", "Champions", "Best customers - high value, frequency, and recent"],
            ["2", "Loyal Customers", "Reliable customers with good metrics"],
            ["3", "Potential Loyalists", "Good customers with potential to improve"],
            ["4", "New Customers", "Recent customers with potential"],
            ["5", "Promising", "New customers with good signs"],
            ["6", "Need Attention", "Average customers requiring attention"],
            ["7", "About to Sleep", "Customers showing decline"],
            ["8", "At Risk", "Important customers at risk of churning"],
            ["9", "Can't Loose", "High-value customers but haven't purchased recently"],
            ["10", "Hibernating", "Lowest value customers"]
        ]

        print(tabulate(hierarchy_explanation,
                       headers=["Rank", "Segment", "Description"],
                       tablefmt="rounded_outline"))

    return segment_analysis


################################################
# CLTV ANALYSIS
################################################

def create_cltv_table(dataframe, customer_id_col="Customer ID",
                      invoice_date_col="InvoiceDate", invoice_col="Invoice",
                      total_price_col="TotalPrice", later_day_analyze=2):
    """
    Create CLTV (Customer Lifetime Value) table.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The input DataFrame.
    customer_id_col : str
        Name of the customer ID column.
    invoice_date_col : str
        Name of the invoice date column.
    invoice_col : str
        Name of the invoice column.
    total_price_col : str
        Name of the total price column.
    later_day_analyze : int
        Number of days for adding to max date in the dataframe so that analyze date determined.

    Returns
    -------
    pandas.DataFrame
        CLTV table with predictions.
    """

    # Calculate analysis date
    analyze_date = dataframe[invoice_date_col].max() + pd.Timedelta(days=later_day_analyze)

    # Calculate CLTV metrics
    cltv = dataframe.groupby(customer_id_col).agg({
        invoice_date_col: [
            lambda x: (x.max() - x.min()).days / 7,  # Recency in weeks
            lambda x: (analyze_date - x.min()).days / 7  # Tenure in weeks
        ],
        invoice_col: lambda x: x.nunique(),  # Frequency
        total_price_col: lambda x: x.sum()  # Monetary
    })

    # Flatten column names
    cltv.columns = ['Recency_CLTV_W', 'Tenure_W', 'Frequency_CLTV', 'Monetary_CLTV']
    cltv = cltv.reset_index()

    # Calculate average monetary value
    cltv["Monetary_CLTV_AVG"] = cltv["Monetary_CLTV"] / cltv["Frequency_CLTV"]

    # Filter positive monetary values
    cltv = cltv[cltv["Monetary_CLTV_AVG"] > 0]

    print("\n\n\n" + "#" * 84)
    print("#" * 30, " " * 5, "CLTV TABLE", " " * 5, "#" * 30)
    print("#" * 84)
    print(cltv)

    return cltv


def fit_bgf_model(cltv_df, penalizer_coef=0.001):
    """
    Fit BG-NBD model for purchase prediction.

    Parameters
    ----------
    cltv_df : pandas.DataFrame
        CLTV DataFrame.
    penalizer_coef : float
        Penalizer coefficient for the model.

    Returns
    -------
    BetaGeoFitter
        Fitted BG-NBD model.
    """
    bgf = BetaGeoFitter(penalizer_coef=penalizer_coef)
    bgf.fit(cltv_df['Frequency_CLTV'],
            cltv_df['Recency_CLTV_W'],
            cltv_df['Tenure_W'])

    return bgf


def fit_ggf_model(cltv_df, penalizer_coef=0.01):
    """
    Fit Gamma-Gamma model for monetary value prediction.

    Parameters
    ----------
    cltv_df : pandas.DataFrame
        CLTV DataFrame.
    penalizer_coef : float
        Penalizer coefficient for the model.

    Returns
    -------
    GammaGammaFitter
        Fitted Gamma-Gamma model.
    """
    ggf = GammaGammaFitter(penalizer_coef=penalizer_coef)
    ggf.fit(cltv_df['Frequency_CLTV'], cltv_df["Monetary_CLTV_AVG"])

    return ggf


def calculate_cltv_predictions(cltv_df, bgf_model, ggf_model, periods=[1, 4, 12, 24, 48]):
    """
    Calculate CLTV predictions for different time periods.

    Parameters
    ----------
    cltv_df : pandas.DataFrame
        CLTV DataFrame.
    bgf_model : BetaGeoFitter
        Fitted BG-NBD model.
    ggf_model : GammaGammaFitter
        Fitted Gamma-Gamma model.
    periods : list
        List of time periods in weeks.

    Returns
    -------
    pandas.DataFrame
        CLTV DataFrame with predictions.
    """
    cltv_result = cltv_df.copy()

    # Create dynamic period names
    period_names = []
    for per in periods:
        if per % 4 == 0 and per % 48 != 0:
            period_names.append(f"{per//4}M")
        elif per % 48 == 0:
            period_names.append(f"{per // 48}Y")
        else:
            period_names.append(f"{per}W")

    # Purchase predictions for different periods
    for period, name in zip(periods, period_names):
        cltv_result[f"Expected_Purc_{name}"] = bgf_model.predict(
            period,
            cltv_result['Frequency_CLTV'],
            cltv_result['Recency_CLTV_W'],
            cltv_result['Tenure_W']
        )

    # Expected average profit
    cltv_result["Expected_Avg_Profit"] = ggf_model.conditional_expected_average_profit(
        cltv_result['Frequency_CLTV'],
        cltv_result["Monetary_CLTV_AVG"]
    )

    # CLTV calculations for different periods
    for period, name in zip(periods, period_names):
        cltv_pred = ggf_model.customer_lifetime_value(
            bgf_model,
            cltv_result['Frequency_CLTV'],
            cltv_result['Recency_CLTV_W'],
            cltv_result['Tenure_W'],
            cltv_result["Monetary_CLTV_AVG"],
            time=period,
            freq="W",
            discount_rate=0.01
        )
        cltv_result[f"CLTV_{name}"] = cltv_pred

    print("\n\n\n" + "#" * 84)
    print("#" * 27, " " * 5, "CLTV PREDICTIONS", " " * 5, "#" * 27)
    print("#" * 84)
    print(cltv_result)

    return cltv_result


def create_cltv_segments(cltv_df, cltv_column="CLTV_6M",
                         cuts=[0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
                         segment_labels=["F", "E", "D", "C", "B", "A"]):
    """
    Create CLTV segments based on quartiles.

    Parameters
    ----------
    cltv_df : pandas.DataFrame
        CLTV DataFrame.
    cltv_column : str
        Column name for CLTV values.
    cuts : list
        Percentage numbers for where to cut segments.
    segment_labels : list
        Labels for the segments.

    Returns
    -------
    pandas.DataFrame
        CLTV DataFrame with segments.
    """
    cltv_result = cltv_df.copy()
    cltv_result["CLTV_Segment"] = pd.qcut(
        cltv_result[cltv_column],
        cuts,
        labels=segment_labels
    )

    print("\n\n\n" + "#" * 84)
    print("#" * 25, " " * 5, "CLTV CLASSIFICATIONS", " " * 5, "#" * 25)
    print("#" * 84)
    print(cltv_result)

    return cltv_result


def analyze_cltv_segments(cltv_df, cltv_columns=["CLTV_1W", "CLTV_1M", "CLTV_3M", "CLTV_6M", "CLTV_1Y"]):
    """
    Analyze CLTV segments and provide comprehensive insights.

    Parameters
    ----------
    cltv_df : pandas.DataFrame
        CLTV DataFrame with segments.
    cltv_columns : list
        List of CLTV column names to analyze.

    Returns
    -------
    pandas.DataFrame
        Segment analysis results.
    """

    print("\n\n\n" + "#" * 84)
    print("#" * 24, " " * 5, "CLTV  SEGMENT ANALYSIS", " " * 5, "#" * 24)
    print("#" * 84)

    # Determine which CLTV columns exist in the dataframe
    existing_cltv_cols = [col for col in cltv_columns if col in cltv_df.columns]

    if not existing_cltv_cols:
        print("⚠️ No CLTV columns found for analysis!")
        return None

    # Create aggregation dictionary dynamically
    agg_dict = {
        'Frequency_CLTV': 'mean',
        'Monetary_CLTV_AVG': 'mean'
    }

    # Add existing CLTV columns
    for col in existing_cltv_cols:
        agg_dict[col] = ['mean', 'count']

    # Perform segment analysis
    segment_analysis = cltv_df.groupby('CLTV_Segment').agg(agg_dict).round(2)

    # Create main segment summary table
    segment_data = []
    for segment in segment_analysis.index:
        row = [segment]

        # Add frequency and monetary averages
        freq_avg = segment_analysis.loc[segment, ('Frequency_CLTV', 'mean')]
        monetary_avg = segment_analysis.loc[segment, ('Monetary_CLTV_AVG', 'mean')]
        row.extend([f"{freq_avg:.1f}", f"{monetary_avg:.2f}"])

        # Add CLTV averages for existing columns
        for col in existing_cltv_cols:
            cltv_avg = segment_analysis.loc[segment, (col, 'mean')]
            row.append(f"{cltv_avg:.2f}")

        # Add customer count (use first CLTV column for count)
        customer_count = int(segment_analysis.loc[segment, (existing_cltv_cols[0], 'count')])
        row.append(customer_count)

        segment_data.append(row)

    # Create headers
    headers = ["Segment", "Avg Frequency", "Avg Monetary ($)"]
    headers.extend([f"Avg {col.replace('CLTV_', '')} CLTV" for col in existing_cltv_cols])
    headers.append("Customer Count")

    print("📊 CLTV SEGMENT SUMMARY:")
    print(tabulate(segment_data, headers=headers, tablefmt="fancy_grid"))

    # Segment distribution with percentages
    segment_counts = cltv_df['CLTV_Segment'].value_counts().sort_index()
    segment_percentages = (segment_counts / len(cltv_df) * 100).round(2)

    distribution_data = []
    for segment in segment_counts.index:
        distribution_data.append([
            segment,
            segment_counts[segment],
            f"{segment_percentages[segment]:.2f}%"
        ])

    print(f"\n📈 SEGMENT DISTRIBUTION:")
    print(tabulate(distribution_data,
                   headers=["Segment", "Customer Count", "Percentage"],
                   tablefmt="fancy_outline"))

    # CLTV insights for each time period
    if len(existing_cltv_cols) > 0:
        print(f"\n💡 CLTV INSIGHTS BY TIME PERIOD:")
        insights_data = []

        for cltv_col in existing_cltv_cols:
            period_name = cltv_col.replace('CLTV_', '')

            # Find highest and lowest value segments for this period
            cltv_by_segment = cltv_df.groupby('CLTV_Segment')[cltv_col].mean().round(2)
            highest_segment = cltv_by_segment.idxmax()
            lowest_segment = cltv_by_segment.idxmin()

            insights_data.append([
                period_name,
                f"Highest: {highest_segment} (${cltv_by_segment[highest_segment]:.2f})",
                f"Lowest: {lowest_segment} (${cltv_by_segment[lowest_segment]:.2f})",
                f"${cltv_df[cltv_col].mean():.2f}"
            ])

        print(tabulate(insights_data,
                       headers=["Period", "Highest Segment", "Lowest Segment", "Overall Average"],
                       tablefmt="rounded_outline"))

    return segment_analysis


def compare_top_customers(cltv_df, cltv_columns=["CLTV_1W", "CLTV_1M", "CLTV_3M", "CLTV_6M", "CLTV_1Y"],
                          top_n=10, customer_id_col="Customer ID"):
    """
    Compare top customers across different CLTV time periods.

    Parameters
    ----------
    cltv_df : pandas.DataFrame
        CLTV DataFrame.
    cltv_columns : list
        List of CLTV column names to compare.
    top_n : int
        Number of top customers to display.
    customer_id_col : str
        Name of the customer ID column.

    Returns
    -------
    dict
        Dictionary containing top customers for each CLTV period.
    """

    print("\n\n\n" + "#" * 84)
    print("#" * 23, " " * 5, "TOP CUSTOMERS COMPARISON", " " * 5, "#" * 23)
    print("#" * 84)

    # Determine which CLTV columns exist in the dataframe
    existing_cltv_cols = [col for col in cltv_columns if col in cltv_df.columns]

    if not existing_cltv_cols:
        print("⚠️ No CLTV columns found for comparison!")
        return {}

    top_customers_dict = {}

    for cltv_col in existing_cltv_cols:
        period_name = cltv_col.replace('CLTV_', '')

        print(f"\n🏆 TOP {top_n} CUSTOMERS BY {period_name}:")
        print("-" * 70)

        # Get top customers for this period
        top_customers = cltv_df.nlargest(top_n, cltv_col)

        # Prepare data for tabulate
        customer_data = []
        for idx, (_, row) in enumerate(top_customers.iterrows(), 1):
            customer_row = [
                idx,  # Rank
                int(row[customer_id_col]),  # Customer ID
                f"{row[cltv_col]:.2f}"  # CLTV value
            ]

            # Add other CLTV values if they exist
            for other_col in existing_cltv_cols:
                if other_col != cltv_col and other_col in cltv_df.columns:
                    customer_row.append(f"{row[other_col]:.2f}")

            # Add segment if available
            if 'CLTV_Segment' in cltv_df.columns:
                customer_row.append(row['CLTV_Segment'])

            customer_data.append(customer_row)

        # Create headers
        headers = ["Rank", "Customer ID", f"{period_name} CLTV"]

        # Add other CLTV period headers
        for other_col in existing_cltv_cols:
            if other_col != cltv_col:
                other_period = other_col.replace('CLTV_', '')
                headers.append(f"{other_period} CLTV")

        # Add segment header if available
        if 'CLTV_Segment' in cltv_df.columns:
            headers.append("Segment")

        print(tabulate(customer_data, headers=headers, tablefmt="fancy_outline"))

        # Summary statistics table
        summary_data = [
            ["Average CLTV", f"${cltv_df[cltv_col].mean():.2f}"],
            ["Median CLTV", f"${cltv_df[cltv_col].median():.2f}"],
            [f"Top {top_n} Average", f"${top_customers[cltv_col].mean():.2f}"],
            ["Maximum CLTV", f"${cltv_df[cltv_col].max():.2f}"],
            ["Minimum CLTV", f"${cltv_df[cltv_col].min():.2f}"],
            ["Standard Deviation", f"${cltv_df[cltv_col].std():.2f}"]
        ]

        print(f"\n📊 {period_name} STATISTICS SUMMARY:")
        print(tabulate(summary_data,
                       headers=["Metric", "Value"],
                       tablefmt="rounded_outline"))

        top_customers_dict[cltv_col] = top_customers

    # Cross-period comparison summary
    if len(existing_cltv_cols) > 1:
        print(f"\n🔄 CROSS-PERIOD COMPARISON SUMMARY:")
        comparison_data = []

        for cltv_col in existing_cltv_cols:
            period_name = cltv_col.replace('CLTV_', '')
            top_customer_id = cltv_df.loc[cltv_df[cltv_col].idxmax(), customer_id_col]
            max_value = cltv_df[cltv_col].max()

            comparison_data.append([
                period_name,
                int(top_customer_id),
                f"${max_value:.2f}",
                f"${cltv_df[cltv_col].mean():.2f}"
            ])

        print(tabulate(comparison_data,
                       headers=["Period", "Top Customer ID", "Max CLTV", "Average CLTV"],
                       tablefmt="fancy_grid"))

    return top_customers_dict


def compare_customer_rankings(cltv_df, base_column="CLTV_1M", compare_columns=["CLTV_6M", "CLTV_1Y"],
                              top_n=20, customer_id_col="Customer ID"):
    """
    Compare how customer rankings change across different CLTV periods.

    Parameters
    ----------
    cltv_df : pandas.DataFrame
        CLTV DataFrame.
    base_column : str
        Base CLTV column for comparison.
    compare_columns : list
        List of CLTV columns to compare against base.
    top_n : int
        Number of top customers to analyze.
    customer_id_col : str
        Name of the customer ID column.

    Returns
    -------
    dict
        Dictionary containing ranking comparison results.
    """

    print(f"\n\n\n" + "#" * 84)
    print("#" * 21, " " * 5, "CUSTOMER  RANKING COMPARISON", " " * 5, "#" * 21)
    print("#" * 84)

    # Check if columns exist
    all_columns = [base_column] + compare_columns
    existing_cols = [col for col in all_columns if col in cltv_df.columns]

    if len(existing_cols) < 2:
        print("⚠️ Not enough CLTV columns found for ranking comparison!")
        return None

    # Get top customers from base column
    top_customers_base = cltv_df.nlargest(top_n, base_column)[customer_id_col].tolist()

    ranking_results = {}

    for compare_col in compare_columns:
        if compare_col not in cltv_df.columns:
            continue

        base_period = base_column.replace('CLTV_', '')
        compare_period = compare_col.replace('CLTV_', '')

        print(f"\n🔄 RANKING CHANGE: {base_period} vs {compare_period}")
        print("-" * 70)

        # Create rankings
        base_ranking = cltv_df[[customer_id_col, base_column]].sort_values(base_column, ascending=False).reset_index(
            drop=True)
        compare_ranking = cltv_df[[customer_id_col, compare_col]].sort_values(compare_col, ascending=False).reset_index(
            drop=True)

        # Add rank columns
        base_ranking[f'{base_period}_Rank'] = range(1, len(base_ranking) + 1)
        compare_ranking[f'{compare_period}_Rank'] = range(1, len(compare_ranking) + 1)

        # Merge rankings
        merged_rankings = base_ranking.merge(
            compare_ranking[[customer_id_col, compare_col, f'{compare_period}_Rank']],
            on=customer_id_col
        )

        # Calculate rank change
        merged_rankings['Rank_Change'] = merged_rankings[f'{base_period}_Rank'] - merged_rankings[
            f'{compare_period}_Rank']

        # Filter for top customers from base period
        top_comparison = merged_rankings[merged_rankings[customer_id_col].isin(top_customers_base)]

        # Sort by rank change (biggest improvements first)
        top_comparison = top_comparison.sort_values('Rank_Change', ascending=False)

        # Prepare data for tabulate
        ranking_data = []
        for _, row in top_comparison.iterrows():
            # Determine rank change symbol and color
            rank_change = row['Rank_Change']
            if rank_change > 0:
                change_symbol = f"{rank_change}"
                change_status = "Improved"
            elif rank_change < 0:
                change_symbol = f"{rank_change}"
                change_status = "Declined"
            else:
                change_symbol = "0"
                change_status = "Same"

            ranking_data.append([
                int(row[customer_id_col]),
                int(row[f'{base_period}_Rank']),
                int(row[f'{compare_period}_Rank']),
                change_symbol,
                change_status,
                f"${row[base_column]:.2f}",
                f"${row[compare_col]:.2f}"
            ])

        headers = [
            "Customer ID",
            f"{base_period} Rank",
            f"{compare_period} Rank",
            "Change",
            "Status",
            f"{base_period} CLTV",
            f"{compare_period} CLTV"
        ]

        print(f"Top {top_n} customers from {base_period} and their ranking in {compare_period}:")
        print(tabulate(ranking_data, headers=headers, tablefmt="fancy_grid"))

        # Summary insights table
        rank_improved = len(top_comparison[top_comparison['Rank_Change'] > 0])
        rank_declined = len(top_comparison[top_comparison['Rank_Change'] < 0])
        rank_same = len(top_comparison[top_comparison['Rank_Change'] == 0])

        # Find biggest movers
        biggest_improver = top_comparison.loc[top_comparison['Rank_Change'].idxmax()]
        biggest_decliner = top_comparison.loc[top_comparison['Rank_Change'].idxmin()]

        summary_data = [
            ["Improved Ranking", f"{rank_improved} customers", f"{(rank_improved / len(top_comparison) * 100):.1f}%"],
            ["Declined Ranking", f"{rank_declined} customers", f"{(rank_declined / len(top_comparison) * 100):.1f}%"],
            ["Same Ranking", f"{rank_same} customers", f"{(rank_same / len(top_comparison) * 100):.1f}%"],
            ["Biggest Improver", f"Customer {int(biggest_improver[customer_id_col])}",
             f"{biggest_improver['Rank_Change']} positions"],
            ["Biggest Decliner", f"Customer {int(biggest_decliner[customer_id_col])}",
             f"{biggest_decliner['Rank_Change']} positions"],
            ["Average Rank Change", f"{top_comparison['Rank_Change'].mean():.1f}", "positions"]
        ]

        print(f"\n📈 RANKING CHANGES SUMMARY:")
        print(tabulate(summary_data,
                       headers=["Metric", "Value", "Details"],
                       tablefmt="rounded_outline"))

        # Top 5 biggest changes (both directions)
        print(f"\n🎯 TOP 5 RANKING CHANGES:")

        # Biggest improvements
        top_improvements = top_comparison.nlargest(5, 'Rank_Change')
        improvement_data = []
        for _, row in top_improvements.iterrows():
            improvement_data.append([
                int(row[customer_id_col]),
                int(row[f'{base_period}_Rank']),
                int(row[f'{compare_period}_Rank']),
                f"{row['Rank_Change']}"
            ])

        print("\n🔝 Biggest Improvements:")
        print(tabulate(improvement_data,
                       headers=["Customer ID", f"{base_period} Rank", f"{compare_period} Rank", "Change"],
                       tablefmt="fancy_outline"))

        # Biggest declines
        top_declines = top_comparison.nsmallest(5, 'Rank_Change')
        decline_data = []
        for _, row in top_declines.iterrows():
            decline_data.append([
                int(row[customer_id_col]),
                int(row[f'{base_period}_Rank']),
                int(row[f'{compare_period}_Rank']),
                f"{row['Rank_Change']}"
            ])

        print("\n📉 Biggest Declines:")
        print(tabulate(decline_data,
                       headers=["Customer ID", f"{base_period} Rank", f"{compare_period} Rank", "Change"],
                       tablefmt="fancy_outline"))

        ranking_results[f"{base_period}_vs_{compare_period}"] = top_comparison

    return ranking_results


################################################
# MAIN ANALYSIS PIPELINE
################################################

def run_complete_analysis(file_path="online_retail_II.xlsx"):
    """
    Run complete RFM and CLTV analysis.

    Parameters
    ----------
    file_path : str
        Path to the Excel file.
    """
    print("\n\n\n" + "#" * 84)
    print("#" * 84)
    print("Starting Complete Customer Analysis...")
    print("#" * 84)
    print("#" * 84)

    # 1. Load and preprocess data
    print("\n\n\n" + "#" * 84)
    print("#" * 84)
    print("1. LOADING AND PREPROCESSING DATA")
    print("#" * 84)
    print("#" * 84)
    df = load_and_preprocess_data(file_path)
    check_df_tabulate(df)

    # 2. RFM Analysis
    print("\n\n\n" + "#" * 84)
    print("#" * 84)
    print("2. RFM ANALYSIS")
    print("#" * 84)
    print("#" * 84)
    rfm = create_rfm_table(df)
    analyze_rfm_segments(rfm)

    # Export loyal customers
    loyal_customers = rfm[rfm['Segment'] == 'loyal_customers']['Customer ID']
    #loyal_customers.to_excel("loyal_customers.xlsx", index=False)
    print(f"\nLoyal customers exported to 'loyal_customers.xlsx' ({len(loyal_customers)} customers)")

    # 3. CLTV Analysis
    print("\n\n\n" + "#" * 84)
    print("#" * 84)
    print("3. CLTV ANALYSIS")
    print("#" * 84)
    print("#" * 84)
    cltv_data = create_cltv_table(df)

    # Fit models
    print("Fitting BG-NBD model...")
    bgf = fit_bgf_model(cltv_data)

    print("Fitting Gamma-Gamma model...")
    ggf = fit_ggf_model(cltv_data)

    # Calculate predictions
    print("Calculating CLTV predictions...")
    cltv_final = calculate_cltv_predictions(cltv_data, bgf, ggf)

    # Create segments
    cltv_final = create_cltv_segments(cltv_final)

    """# Analyze CLTV segments
    print("\n\n\n" + "#" * 84)
    print("#" * 84)
    print("CLTV SEGMENT ANALYSIS")
    print("#" * 84)
    print("#" * 84)



    segment_analysis = cltv_final.groupby('CLTV_Segment').agg({
        'CLTV_6M': ['mean', 'count'],
        'Frequency_CLTV': 'mean',
        'Monetary_CLTV_AVG': 'mean'
    }).round(2)
    print(segment_analysis)

    # Compare top customers
    print("\n4. TOP CUSTOMERS COMPARISON")
    print("Top 10 customers by 1-month CLTV:")
    print(cltv_final.nlargest(10, 'CLTV_1M')[['Customer ID', 'CLTV_1M', 'CLTV_6M', 'CLTV_1Y']])

    print("\nTop 10 customers by 1-year CLTV:")
    print(cltv_final.nlargest(10, 'CLTV_1Y')[['Customer ID', 'CLTV_1M', 'CLTV_6M', 'CLTV_1Y']])
    """

    # 4. CLTV Segment Analysis
    analyze_cltv_segments(cltv_final)

    # 5. Top Customers Comparison
    top_customers_results = compare_top_customers(cltv_final, top_n=10)

    # 6. Customer Ranking Changes
    ranking_changes = compare_customer_rankings(cltv_final,
                                                base_column="CLTV_1M",
                                                compare_columns=["CLTV_6M", "CLTV_1Y"],
                                                top_n=15)
    # Visualization
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    rfm['Segment'].value_counts().plot(kind='bar')
    plt.title('RFM Segment Distribution')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 2)
    cltv_final['CLTV_Segment'].value_counts().plot(kind='bar')
    plt.title('CLTV Segment Distribution')

    plt.subplot(2, 2, 3)
    plt.hist(cltv_final['CLTV_6M'], bins=50, alpha=0.7)
    plt.title('CLTV 6-Month Distribution')
    plt.xlabel('CLTV Value')

    plt.subplot(2, 2, 4)
    plot_period_transactions(bgf)
    plt.title('BG-NBD Model Validation')

    plt.tight_layout()
    plt.show(block=True)

    return df, rfm, cltv_final, top_customers_results, ranking_changes


################################################
# EXECUTION
################################################

if __name__ == "__main__":
    # Run the complete analysis
    df, rfm_results, cltv_results, top_customers, rankings = run_complete_analysis()

    print("\n\n\n" + "#" * 84)
    print("Analysis completed successfully!")
    print(f"RFM results shape: {rfm_results.shape}")
    print(f"CLTV results shape: {cltv_results.shape}")
    print(f"Top customers analysis completed for {len(top_customers)} periods")
    print(f"Ranking comparisons completed for {len(rankings)} period pairs")
    print("#" * 84)