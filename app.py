
import warnings

import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="Data Visualization Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 General-Purpose Data Visualization Dashboard")
st.caption(
    "Upload any CSV file, explore its data, and build interactive visualizations."
)


# -------------------- Helper functions --------------------

def get_column_types(df):
    """Identify numeric and categorical columns without assuming a schema."""
    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        series = df[col]

        if pd.api.types.is_bool_dtype(series):
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
        else:
            # Convert mixed-type columns to safe display/grouping labels.
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def get_category_columns(df):
    """Return columns that can safely be used as categories."""
    return [col for col in df.columns if col in df.columns]


def clean_numeric(series):
    """Convert a series to numeric; invalid values become missing."""
    return pd.to_numeric(series, errors="coerce")


def unique_labels(series):
    """Convert mixed values to consistent, readable categorical labels."""
    return series.astype("string").fillna("(Missing)").astype(str)


def add_axis_range_controls(axis, series):
    """Optionally apply a custom range to a numeric axis."""
    enabled = st.sidebar.checkbox(
        f"Set custom {axis.upper()} range",
        key=f"range_{axis}_{series.name}",
    )

    if not enabled:
        return None

    values = clean_numeric(series).dropna()

    if values.empty:
        st.sidebar.warning(f"No valid numeric values for the {axis.upper()} axis.")
        return None

    minimum = float(values.min())
    maximum = float(values.max())

    if minimum == maximum:
        padding = max(abs(minimum) * 0.05, 1.0)
        minimum -= padding
        maximum += padding

    low, high = st.sidebar.slider(
        f"{axis.upper()}-axis limits",
        min_value=minimum,
        max_value=maximum,
        value=(minimum, maximum),
        key=f"slider_{axis}_{series.name}",
    )

    return [low, high]


def apply_axis_range(fig, axis, axis_range):
    if axis_range is not None:
        if axis == "x":
            fig.update_xaxes(range=axis_range)
        else:
            fig.update_yaxes(range=axis_range)


def optional_category(label, options, key):
    """Provide an optional category selector."""
    choices = ["None"] + list(options)
    return st.sidebar.selectbox(label, choices, key=key)


def show_questions(chart_type):
    """Display interpretation prompts relevant to the selected chart."""
    questions = {
        "Histogram": [
            "What is the shape of the distribution?",
            "Are there potential outliers?",
            "Is the distribution symmetric or skewed?",
        ],
        "Bar chart": [
            "Which categories are largest or smallest?",
            "Are the observed differences economically meaningful?",
            "What business decision might this comparison inform?",
        ],
        "Box plot": [
            "Do the groups differ in median?",
            "Which groups show more variability?",
            "Are there notable outliers?",
        ],
        "Scatter plot": [
            "Is there an apparent relationship between X and Y?",
            "Is the relationship approximately linear?",
            "Are there clusters or unusual observations?",
            "Do color or size reveal additional patterns?",
        ],
    }

    with st.expander("Questions to consider", expanded=False):
        for question in questions[chart_type]:
            st.markdown(f"- {question}")


# -------------------- File upload --------------------

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file",
    type=["csv"],
)

if uploaded_file is None:
    st.info("Upload a CSV file using the sidebar to get started.")
    st.markdown(
        """
        **Available visualizations**
        - Histogram: inspect distributions.
        - Bar chart: compare categories and aggregated metrics.
        - Box plot: compare distributions across groups.
        - Scatter plot: explore relationships between numeric variables.
        """
    )
    st.stop()

try:
    # Preserve the uploaded file and avoid assuming any column names.
    df = pd.read_csv(uploaded_file, low_memory=False)
except Exception as exc:
    st.error(f"Unable to read this CSV file. Check its format. Details: {exc}")
    st.stop()

if df.empty or len(df.columns) == 0:
    st.error("The uploaded CSV contains no usable rows or columns.")
    st.stop()

# Remove surrounding whitespace from column names and make duplicates unique.
df.columns = [str(col).strip() or f"Column_{i+1}"
              for i, col in enumerate(df.columns)]

if df.columns.duplicated().any():
    counts = {}
    new_columns = []

    for col in df.columns:
        counts[col] = counts.get(col, 0) + 1
        new_columns.append(
            col if counts[col] == 1 else f"{col}_{counts[col]}"
        )

    df.columns = new_columns

numeric_cols, categorical_cols = get_column_types(df)
all_cols = list(df.columns)

# Overview metrics.
st.subheader("Dataset overview")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{len(df):,}")
m2.metric("Columns", f"{len(df.columns):,}")
m3.metric("Numeric columns", len(numeric_cols))
m4.metric("Categorical columns", len(categorical_cols))

with st.expander("Column classification", expanded=False):
    st.write("**Numeric columns**")
    st.write(numeric_cols or "No numeric columns detected.")

    st.write("**Categorical/text columns**")
    st.write(categorical_cols or "No categorical columns detected.")

    st.write("**Missing values by column**")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        st.write("No missing values detected.")
    else:
        st.dataframe(
            missing.rename("Missing values").to_frame(),
            use_container_width=True,
        )

st.subheader("Data preview")
st.dataframe(df.head(10), use_container_width=True)


# -------------------- Chart selection --------------------

chart_type = st.sidebar.selectbox(
    "Choose chart type",
    ["Histogram", "Bar chart", "Box plot", "Scatter plot"],
)

st.subheader(chart_type)

fig = None
error_message = None

# -------------------- Histogram --------------------

if chart_type == "Histogram":
    if not numeric_cols:
        st.warning("A histogram requires at least one numeric column.")
        st.stop()

    x_col = st.sidebar.selectbox("Numeric variable", numeric_cols)

    bins = st.sidebar.slider(
        "Number of bins", min_value=5, max_value=100, value=30
    )
    opacity = st.sidebar.slider(
        "Opacity", min_value=0.1, max_value=1.0, value=0.75, step=0.05
    )

    color_col = optional_category(
        "Color/group by", categorical_cols, "hist_color"
    )

    x_range = add_axis_range_controls("x", df[x_col])

    plot_df = pd.DataFrame({"value": clean_numeric(df[x_col])})
    plot_df["value"] = plot_df["value"].replace(
        [float("inf"), float("-inf")], float("nan")
    )

    if color_col != "None":
        plot_df["group"] = unique_labels(df[color_col])

    plot_df = plot_df.dropna(subset=["value"])

    if plot_df.empty:
        st.warning("The selected column contains no valid numeric values.")
        st.stop()

    fig = px.histogram(
        plot_df,
        x="value",
        color="group" if color_col != "None" else None,
        nbins=bins,
        opacity=opacity,
        marginal="rug",
        barmode="overlay" if color_col != "None" else "relative",
        labels={"value": x_col, "group": color_col},
        title=f"Distribution of {x_col}",
        hover_data=["group"] if color_col != "None" else None,
    )
    apply_axis_range(fig, "x", x_range)
    fig.update_yaxes(title="Number of observations")


# -------------------- Bar chart --------------------

elif chart_type == "Bar chart":
    if not categorical_cols:
        st.warning(
            "A bar chart requires a categorical/text column. "
            "This dataset has none detected."
        )
        st.stop()

    x_col = st.sidebar.selectbox(
        "Category (X-axis)", categorical_cols
    )

    metric_options = ["Count of observations"] + numeric_cols
    metric = st.sidebar.selectbox("Measure", metric_options)

    if metric == "Count of observations":
        aggregation = "count"
        metric_col = None
    else:
        metric_col = metric
        aggregation = st.sidebar.selectbox(
            "Aggregation function",
            ["mean", "median", "sum", "count"],
        )

    color_col = optional_category(
        "Color/group by",
        [c for c in categorical_cols if c != x_col],
        "bar_color",
    )

    bar_mode = st.sidebar.radio(
        "Bar arrangement", ["Grouped", "Stacked"], horizontal=True
    )

    sort_values = st.sidebar.checkbox(
        "Sort categories by plotted value", value=False
    )

    working = pd.DataFrame({
        "category": unique_labels(df[x_col]),
    })

    if color_col != "None":
        working["group"] = unique_labels(df[color_col])

    if metric_col is not None:
        working["measure"] = clean_numeric(df[metric_col])
        working["measure"] = working["measure"].replace(
            [float("inf"), float("-inf")], float("nan")
        )

    group_keys = ["category"]
    if color_col != "None":
        group_keys.append("group")

    if metric_col is None:
        summary = working.groupby(
            group_keys, dropna=False
        ).size().reset_index(name="value")
        y_title = "Number of observations"
    else:
        if aggregation == "count":
            summary = working.groupby(
                group_keys, dropna=False
            )["measure"].count().reset_index(name="value")
        else:
            summary = working.groupby(
                group_keys, dropna=False
            )["measure"].agg(aggregation).reset_index(name="value")

        y_title = f"{aggregation.title()} of {metric_col}"

    if summary.empty:
        st.warning("No data is available to plot.")
        st.stop()

    if sort_values:
        summary = summary.sort_values("value", ascending=False)

    fig = px.bar(
        summary,
        x="category",
        y="value",
        color="group" if color_col != "None" else None,
        barmode="group" if bar_mode == "Grouped" else "stack",
        labels={
            "category": x_col,
            "value": y_title,
            "group": color_col,
        },
        title=f"{y_title} by {x_col}",
        hover_data=["value"],
    )

    fig.update_xaxes(title=x_col)
    fig.update_yaxes(title=y_title)


# -------------------- Box plot --------------------

elif chart_type == "Box plot":
    if not numeric_cols:
        st.warning("A box plot requires at least one numeric column.")
        st.stop()

    y_col = st.sidebar.selectbox("Numeric variable", numeric_cols)

    group_col = optional_category(
        "Compare groups (X-axis)", categorical_cols, "box_group"
    )

    available_colors = [
        c for c in categorical_cols if c != group_col
    ] if group_col != "None" else categorical_cols

    color_col = optional_category(
        "Additional color grouping", available_colors, "box_color"
    )

    points_option = st.sidebar.selectbox(
        "Show data points",
        ["None", "Outliers only", "All points"],
    )

    points_map = {
        "None": False,
        "Outliers only": "outliers",
        "All points": "all",
    }

    y_range = add_axis_range_controls("y", df[y_col])

    plot_df = pd.DataFrame({
        "value": clean_numeric(df[y_col])
    })
    plot_df["value"] = plot_df["value"].replace(
        [float("inf"), float("-inf")], float("nan")
    )

    if group_col != "None":
        plot_df["group"] = unique_labels(df[group_col])

    if color_col != "None":
        plot_df["color_group"] = unique_labels(df[color_col])

    plot_df = plot_df.dropna(subset=["value"])

    if plot_df.empty:
        st.warning("The selected column contains no valid numeric values.")
        st.stop()

    fig = px.box(
        plot_df,
        x="group" if group_col != "None" else None,
        y="value",
        color="color_group" if color_col != "None" else (
            "group" if group_col != "None" else None
        ),
        points=points_map[points_option],
        labels={
            "group": group_col,
            "value": y_col,
            "color_group": color_col,
        },
        title=f"Distribution of {y_col}",
    )

    if group_col != "None":
        fig.update_xaxes(title=group_col)
    else:
        fig.update_xaxes(title="Overall distribution")

    fig.update_yaxes(title=y_col)
    apply_axis_range(fig, "y", y_range)


# -------------------- Scatter plot --------------------

elif chart_type == "Scatter plot":
    if len(numeric_cols) < 2:
        st.warning(
            "A scatter plot requires at least two numeric columns."
        )
        st.stop()

    x_col = st.sidebar.selectbox("X-axis variable", numeric_cols)
    y_options = [c for c in numeric_cols if c != x_col] or numeric_cols
    y_col = st.sidebar.selectbox("Y-axis variable", y_options)

    color_options = all_cols
    color_col = optional_category(
        "Color by", color_options, "scatter_color"
    )

    size_col = optional_category(
        "Point size (numeric)", numeric_cols, "scatter_size"
    )

    symbol_col = optional_category(
        "Marker symbol", categorical_cols, "scatter_symbol"
    )

    hover_cols = st.sidebar.multiselect(
        "Additional hover information",
        [c for c in all_cols if c not in [x_col, y_col]],
        default=[],
    )

    opacity = st.sidebar.slider(
        "Point opacity",
        min_value=0.1,
        max_value=1.0,
        value=0.8,
        step=0.05,
    )

    x_range = add_axis_range_controls("x", df[x_col])
    y_range = add_axis_range_controls("y", df[y_col])

    trendline = st.sidebar.checkbox("Add trend line", value=False)

    plot_df = pd.DataFrame({
        "x_value": clean_numeric(df[x_col]),
        "y_value": clean_numeric(df[y_col]),
    })

    plot_df["x_value"] = plot_df["x_value"].replace(
        [float("inf"), float("-inf")], float("nan")
    )
    plot_df["y_value"] = plot_df["y_value"].replace(
        [float("inf"), float("-inf")], float("nan")
    )

    if color_col != "None":
        if color_col in numeric_cols and color_col not in [x_col, y_col]:
            plot_df["color_value"] = clean_numeric(df[color_col])
        else:
            plot_df["color_value"] = unique_labels(df[color_col])

    if size_col != "None":
        plot_df["size_value"] = clean_numeric(df[size_col])
        plot_df["size_value"] = plot_df["size_value"].clip(lower=0)

    if symbol_col != "None":
        plot_df["symbol_value"] = unique_labels(df[symbol_col])

    hover_mapping = {}
    for i, col in enumerate(hover_cols):
        key = f"hover_{i}"
        plot_df[key] = df[col].astype("string").fillna("(Missing)")
        hover_mapping[key] = col

    plot_df = plot_df.dropna(subset=["x_value", "y_value"])

    if plot_df.empty:
        st.warning("No rows contain valid numeric X and Y values.")
        st.stop()

    if trendline and plot_df["x_value"].nunique() < 2:
        st.warning("A trend line requires at least two distinct X values.")
        trendline = False

    # OLS trend lines require statsmodels.
    if trendline:
        try:
            import statsmodels.api  # noqa: F401
        except ImportError:
            st.error(
                "Trend lines require statsmodels. Install it using "
                "`pip install statsmodels` or install requirements.txt."
            )
            trendline = False

    try:
        fig = px.scatter(
            plot_df,
            x="x_value",
            y="y_value",
            color="color_value" if color_col != "None" else None,
            size="size_value" if size_col != "None" else None,
            symbol="symbol_value" if symbol_col != "None" else None,
            hover_data=hover_mapping or None,
            opacity=opacity,
            trendline="ols" if trendline else None,
            labels={
                "x_value": x_col,
                "y_value": y_col,
                "color_value": color_col,
                "size_value": size_col,
                "symbol_value": symbol_col,
            },
            title=f"{y_col} vs. {x_col}",
        )
    except Exception as exc:
        st.error(f"Could not create this scatter plot: {exc}")
        st.stop()

    apply_axis_range(fig, "x", x_range)
    apply_axis_range(fig, "y", y_range)


# -------------------- Render visualization --------------------

if fig is not None:
    fig.update_layout(
        template="plotly_white",
        legend_title_text="",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)
    show_questions(chart_type)
