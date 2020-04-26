import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing
from sklearn.decomposition import PCA

if __name__ == '__main__':
    # *************************************** #
    # *************************************** #
    # LOADING DATA FROM CSV INTO DATAFRAME USING PANDAS
    dataframe = pd.read_csv("data/application_train.csv")

    # *************************************** #
    # *************************************** #
    # HANDLING CATEGORICAL ATTRIBUTES
    # using one hot encoding for binary attributes
    cat_columns = ["NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
                   "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_FAMILY_STATUS",
                   "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "WEEKDAY_APPR_PROCESS_START", "ORGANIZATION_TYPE",
                   "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE"]

    df = pd.get_dummies(dataframe, prefix_sep="_", columns=cat_columns)

    # using ordinal mapping for education
    df["NAME_EDUCATION_TYPE"] = df["NAME_EDUCATION_TYPE"].map({
        'Lower secondary': 0,
        'Secondary / secondary special': 1,
        'Incomplete higher': 2,
        'Higher education': 3,
        'Academic degree': 4
    })

    # *************************************** #
    # *************************************** #
    # remove id column and store it, no point to apply preprocessing on this column
    df_ids = df["SK_ID_CURR"]
    df_without_ids = df.drop("SK_ID_CURR", axis=1)

    # *************************************** #
    # *************************************** #
    # HANDLING MISSING VALUES
    # missing values with median per column
    for col in df_without_ids:
        median = df_without_ids[col].median(axis=0)
        df_without_ids[col] = df_without_ids[col].fillna(median)

    # missing values with bayesian ridge
    # bayesian_ridge_imputer = IterativeImputer()
    #
    # columns = list(df)
    # df = bayesian_ridge_imputer.fit_transform(df)
    #
    # df = pd.DataFrame(data=df, columns=columns)

    # *************************************** #
    # *************************************** #
    # STANDARDIZATION
    # custom standardization
    # for col in df_without_ids:
    #     mean = df_without_ids[col].mean()
    #     sd = df_without_ids[col].std()
    #
    #     df_without_ids[col] = (df_without_ids[col] - mean) / sd

    # by using scale method of sklearn.preprocessing
    # columns = list(df_without_ids)
    # df_without_ids = preprocessing.scale(df_without_ids)
    #
    # df_without_ids = pd.DataFrame(data=df_without_ids,columns=columns)

    # *************************************** #
    # *************************************** #
    # NORMALIZATION
    # custom normalization
    # for col in df_without_ids:
    #     df_max = df_without_ids[col].max()
    #     df_min = df_without_ids[col].min()
    #
    #     df_without_ids[col] = (df_without_ids[col] - df_min) / (df_max - df_min)

    # by using normalize method of sklearn.preprocessing
    columns = list(df_without_ids)
    df_without_ids = preprocessing.normalize(df_without_ids, axis=0)
    df_without_ids = pd.DataFrame(df_without_ids, columns=columns)

    # *************************************** #
    # *************************************** #
    # PCA ANALYSIS
    pca = PCA(n_components=50)

    principal_components = pca.fit_transform(df_without_ids)

    columns = ["principal_component_" + str(i) for i in range(1, 51)]

    pca_df = pd.DataFrame(principal_components, columns=columns)

    # adding final ID attribute
    pca_df["ID"] = df_ids

    # ADDITIONAL LOGIC TO BE IMPLEMENTED
