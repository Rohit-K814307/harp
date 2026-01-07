import pandas as pd


code_columns_dgns = ['ADMTNG_ICD9_DGNS_CD','ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2',
               'ICD9_DGNS_CD_3', 'ICD9_DGNS_CD_4', 'ICD9_DGNS_CD_5', 'ICD9_DGNS_CD_6',
               'ICD9_DGNS_CD_7', 'ICD9_DGNS_CD_8', 'ICD9_DGNS_CD_9', 'ICD9_DGNS_CD_10']

procedure_columns = ['ICD9_PRCDR_CD_1', 'ICD9_PRCDR_CD_2', 'ICD9_PRCDR_CD_3',
                     'ICD9_PRCDR_CD_4', 'ICD9_PRCDR_CD_5', 'ICD9_PRCDR_CD_6']
               


MISSING_DGNS_SENTINEL = -1
OTHER_DGNS_SENTINEL = 300000
INVALID_DGNS_SENTINEL = 400000

MISSING_PRCDR_SENTINEL = -1
OTHER_PRCDR_SENTINEL = 300000
INVALID_PRCDR_SENTINEL = 400000

modes = {
    "DGNS":[MISSING_DGNS_SENTINEL, OTHER_DGNS_SENTINEL, INVALID_DGNS_SENTINEL],
    "PRCDR":[MISSING_PRCDR_SENTINEL, OTHER_PRCDR_SENTINEL, INVALID_PRCDR_SENTINEL]
}

CONSTANTS_DGNS = {
    "V":10000,
    "E":20000
}

CONSTANTS_PRCDR = {
    "V":110000,
    "E":120000
}


def map_code_to_int(value, constants, mode="DGNS"):
    
    MISSING_SENTINEL, OTHER_SENTINEL, INVALID_SENTINEL = modes[mode]

    # Handle None/NaN
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return MISSING_SENTINEL

    s = str(value).strip()
    if s == '' or s.upper() == 'NONE' or s.upper() == 'MISSING':
        return MISSING_SENTINEL

    # Try plain numeric
    try:
        return int(float(s))
    except (ValueError, TypeError):
        pass

    upper = s.upper()
    # V codes
    if upper.startswith('V') and upper[1:].replace('.', '', 1).isdigit():
        try:
            return int(float(upper[1:])) + constants["V"]
        except Exception:
            return INVALID_SENTINEL
    # E codes
    if upper.startswith('E') and upper[1:].replace('.', '', 1).isdigit():
        try:
            return int(float(upper[1:])) + constants["E"]
        except Exception:
            return INVALID_SENTINEL

    # Other non-numeric strings -> 300000
    if upper == 'OTH':
        print(upper)
        return OTHER_SENTINEL

    # Fallback invalid
    return INVALID_SENTINEL


def encode_code_columns(df, cols, constants, mode):
    df_out = df.copy()
    for col in cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].apply(lambda x: map_code_to_int(x, constants, mode))
    return df_out
