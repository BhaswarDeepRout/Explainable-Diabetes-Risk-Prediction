import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

def preprocess_with_columntransformer(df, target_col="diabetes", random_state=42, test_size=0.2):
    # Determine the train indices exactly as downstream will do
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col]
    )
    
    numerical_columns = df.drop(columns=[target_col]).select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_columns = df.drop(columns=[target_col]).select_dtypes(include=["object", "category"]).columns.tolist()
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median"))
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('encoder', OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_columns),
            ('cat', cat_pipeline, categorical_columns)
        ],
        remainder='passthrough'
    )
    
    # Fit only on training data
    preprocessor.fit(df.loc[train_idx])
    
    # Transform full data
    transformed = preprocessor.transform(df)
    
    # Get feature names
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_columns)
    
    # Remainder is the target column
    feature_names = numerical_columns + list(cat_feature_names) + [target_col]
    
    df_transformed = pd.DataFrame(transformed, columns=feature_names, index=df.index)
    
    # Convert data types back appropriately
    for col in numerical_columns:
        df_transformed[col] = df_transformed[col].astype(float) # or int
    df_transformed[target_col] = df_transformed[target_col].astype(int)
    
    for col in cat_feature_names:
        df_transformed[col] = df_transformed[col].astype(int)
        
    return df_transformed, preprocessor
