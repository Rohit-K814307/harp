import numpy as np
import pandas as pd


### make the dataset clean for dataset

def clean_df(df):

     mandatory_cols = [
          'DESYNPUF_ID', 'CLM_ID', 'PRVDR_NUM', 'AT_PHYSN_NPI', 
          'CLM_ADMSN_DT', 'NCH_BENE_DSCHRG_DT',
          'CLM_PMT_AMT', 'CLM_DRG_CD', 'ADMTNG_ICD9_DGNS_CD',
          'CLM_UTLZTN_DAY_CNT'
     ]
     
     optional_cols = ['OP_PHYSN_NPI'] + \
                     [f'ICD9_DGNS_CD_{i}' for i in range(1, 11)] + \
                     [f'ICD9_PRCDR_CD_{i}' for i in range(1, 7)]
     
     financial_cols = [
          'NCH_PRMRY_PYR_CLM_PD_AMT', 'NCH_BENE_IP_DDCTBL_AMT', 
          'NCH_BENE_PTA_COINSRNC_LBLTY_AM'
     ]

     # keep relevant columns
     final_cols = mandatory_cols + optional_cols + financial_cols
     df = df[final_cols].copy()
     print(f"Initial size: {len(df)}")

     # drop rows missing important things only
     df.dropna(subset=mandatory_cols, inplace=True)

     # fill optional clinical and financial fields
     df[optional_cols] = df[optional_cols].fillna("NONE")
     df[financial_cols] = df[financial_cols].fillna(0.0)

     print(f"Final clean size: {len(df)}")
     
     df.to_csv("harp/data/raw/cms_2008_2010_samples_cleaned.csv", index=False)


### group by 3 types of reviewer; lower level, mid level, senior level

def group_by_reviewer(df):
     
     def calculate_complexity(row):
          score = 0
          reasons = []

          if row['NCH_PRMRY_PYR_CLM_PD_AMT'] > 0:
               score += 3
               reasons.append("Multi-Payer")

          if row['CLM_UTLZTN_DAY_CNT'] <= 1:
               score += 3
               reasons.append("Short-Stay")

          op_phys = row['OP_PHYSN_NPI']
          if pd.notna(op_phys) and str(op_phys).upper() != "MISSING":
               score += 2
               reasons.append("Surgical")

          if row['CLM_PMT_AMT'] > 20000:
               score += 1
               reasons.append("High-Cost")

          if score >= 5:
               tier = 3
          elif score >= 2:
               tier = 2
          else:
               tier = 1

          return pd.Series([score, tier, reasons])

     df[['reviewer_score', 'reviewer', 'reviewer_reasons']] = df.apply(calculate_complexity, axis=1)
     df.to_csv("harp/data/raw/csm_2008_2010_samples_grouped.csv", index=False)



### inject synthetic failed claims

def corrupt_accepted(chunk, reason_type):
    
    # copy given dataset chunk
    chunk = chunk.copy()
    
    # set labels
    chunk['claim_status'] = 0
    chunk['rejection_reason_code'] = reason_type
    
    # discharge date precedes admission date by subtracting 500 from integer date to ensure logic breaks
    if reason_type == 'date_mismatch':
        chunk['NCH_BENE_DSCHRG_DT'] = chunk['CLM_ADMSN_DT'] - 500 
        chunk['rejection_desc'] = "Discharge Date precedes Admission Date"

    # missing attending physician; use "missing" string instead of nan so embeddings pick it up
    elif reason_type == 'missing_physician':
        chunk['AT_PHYSN_NPI'] = "MISSING"
        chunk['rejection_desc'] = "Missing Attending Physician NPI"

    # short stay (1 day) but billed high; set utilization to 1 and sync dates to simulate medically unnecessary admission
    elif reason_type == 'level_of_care':
        chunk['CLM_UTLZTN_DAY_CNT'] = 1
        chunk['NCH_BENE_DSCHRG_DT'] = chunk['CLM_ADMSN_DT'] 
        chunk['rejection_desc'] = "Inpatient admission not medically necessary (Short Stay)"

    # financial impossibility where deductible (5000) exceeds total payment (1000)
    elif reason_type == 'financial_error':
        chunk['NCH_BENE_IP_DDCTBL_AMT'] = 5000.00
        chunk['CLM_PMT_AMT'] = 1000.00
        chunk['rejection_desc'] = "Deductible amount exceeds Total Payment amount"

    # garbage diagnosis code; set admitting diagnosis to "invalid_999"
    elif reason_type == 'invalid_code':
        chunk['ADMTNG_ICD9_DGNS_CD'] = "INVALID_999"
        chunk['rejection_desc'] = "Invalid or Missing Admitting Diagnosis Code"

    # missing drg code; use "missing" string instead of nan to indicate critical data failure
    elif reason_type == 'missing_drg':
        chunk['CLM_DRG_CD'] = "MISSING"
        chunk['rejection_desc'] = "Claim submitted without DRG Code"

    return chunk

def inject_rejected(df, reject_ratio):
     
     reject_reasons = [
          'date_mismatch', 'missing_physician', 'level_of_care',
          'financial_error', 'invalid_code', 'missing_drg'
     ]
    
     n_reject = int(len(df) * reject_ratio)

     #set up important cols
     df_accepted = df.copy()
     df_accepted["claim_status"] = 1
     df_accepted["rejection_reason_code"] = "None"
     df_accepted["rejection_desc"] = "Accepted"

     df_to_reject = df.iloc[:n_reject].sample(frac=1).reset_index(drop=True)

     rng = np.random.default_rng()
     reject_chunk_parts = rng.uniform(low=0.2, high=1.0, size=len(reject_reasons))
     normalized = reject_chunk_parts / reject_chunk_parts.sum()
     chunk_sizes = (normalized * n_reject).astype(int)
     chunk_sizes[0] += n_reject - chunk_sizes.sum()

     rejected_chunks = []
     j = 0
          
     for i, reason in enumerate(reject_reasons):
          count = chunk_sizes[i]
          k = j + count

          sliced = df_to_reject.iloc[j:k]
          if not sliced.empty:
              rejected_chunks.append(corrupt_accepted(sliced, reason))

          j = k
     
     return pd.concat([df_accepted, pd.concat(rejected_chunks)]).sample(frac=1).reset_index(drop=True)



### Encode categorical variables

def encode(df):

     # list of categorical vars
     cats = []





