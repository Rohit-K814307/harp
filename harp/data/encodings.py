import urllib.request
import os
import shutil
import pandas as pd

PRCDR_DROP = ['ICD9_PRCDR_CD_2', 'ICD9_PRCDR_CD_3', 'ICD9_PRCDR_CD_4', 'ICD9_PRCDR_CD_5', 'ICD9_PRCDR_CD_6']

code_cols_dgns = ['ADMTNG_ICD9_DGNS_CD','ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2',
               'ICD9_DGNS_CD_3', 'ICD9_DGNS_CD_4', 'ICD9_DGNS_CD_5', 'ICD9_DGNS_CD_6',
               'ICD9_DGNS_CD_7', 'ICD9_DGNS_CD_8', 'ICD9_DGNS_CD_9', 'ICD9_DGNS_CD_10']

code_cols_prcdr = ['ICD9_PRCDR_CD_1']


def download_vocab():
    url = "https://www.cms.gov/medicare/coding/icd9providerdiagnosticcodes/downloads/cmsv28_master_descriptions.zip"
    save_path = "harp/data/raw/icd9_vocab.zip"

    urllib.request.urlretrieve(url, save_path)

    shutil.unpack_archive(save_path, "harp/data/raw/icd9_vocab")
    os.remove(save_path)


def txt_to_vocab(mode):
    form = "SG"
    if mode == "DGNS":
        form = "DX"

    file = f"harp/data/raw/icd9_vocab/CMS28_DESC_LONG_SHORT_{form}.xls"
    df = pd.read_excel(file)
    df.insert(loc=0, column='CODING', value=df.index+1)

    if mode == "DGNS":
        df.rename(columns={"DIAGNOSIS CODE": "ICD9"}, inplace=True)
    else:
        df.rename(columns={"PROCEDURE CODE": "ICD9"}, inplace=True)

    missing_data = {
        "ICD9":["MISSING", "OTH", "INVALID_999"],
        "LONG DESCRIPTION": ["missing val", "other val", "invalid val"],
        "SHORT DESCRIPTION": ["missing val", "other val", "invalid val"],
        "CODING": [-1, 0, -2]
    }

    return pd.concat([df, pd.DataFrame(missing_data)], ignore_index=True)
    


def get_vocab():
    diagnosis_df = txt_to_vocab("DGNS")
    prcdr_df = txt_to_vocab("PRCDR")


    mode_to_vocab = {
        "DGNS":diagnosis_df,
        "PRCDR":prcdr_df,
    }

    return mode_to_vocab


def get_vocab_size():
    vocab = get_vocab()
    
    return {
        "DGNS":len(vocab["DGNS"].keys()),
        "PRCDR":len(vocab["PRCDR"].keys())
    }


def encode_code_columns(df, cols, mode):
    df_out = df.copy()
    df_out[code_cols_prcdr] = df_out[code_cols_prcdr].astype(str).replace(r'\.0$', '', regex=True)
    df_out[code_cols_dgns] = df_out[code_cols_dgns].astype(str)


    vocab_df = get_vocab()[mode]

    code_map = dict(zip(vocab_df["ICD9"].astype(str), vocab_df["CODING"]))

    for col in cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].map(code_map).fillna(-1).astype(int)
            
    return df_out
