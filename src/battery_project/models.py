"""The three compared regressors, each wrapped with its preprocessing in an sklearn Pipeline."""
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

MODEL_NAMES = ["LinearRegression", "DecisionTree", "RandomForest"]


def make_pipeline(name, random_state, large_data=False):
    """Fixed, modest settings (no tuning). `large_data` uses larger leaves for the SOC sample table."""
    leaf = 20 if large_data else 3
    if name == "LinearRegression":
        model = LinearRegression()
    elif name == "DecisionTree":
        model = DecisionTreeRegressor(max_depth=8, min_samples_leaf=leaf, random_state=random_state)
    elif name == "RandomForest":
        model = RandomForestRegressor(n_estimators=100 if large_data else 300, min_samples_leaf=leaf,
                                      max_depth=16 if large_data else None,
                                      random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(name)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])
