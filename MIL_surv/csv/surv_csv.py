import pandas as pd
import os
import argparse
import re    
from sklearn.preprocessing import LabelEncoder

## Survival

os_bin_labels = {'Dead':1,'Alive':0}
rpfs_bin_labels = {'Radiographic prog. or death':1, 'Alive without radiographic prog.':0}

bin_labels = {'os':os_bin_labels, 'rpfs': rpfs_bin_labels}
#--->Setting parameters
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_files', type=str)
    parser.add_argument('--path_clinical_csv', type=str)
    parser.add_argument('--path_marker_slides_csv', type=str, default= '/gpfs/workdir/chrakii/data_info/marker_slides.csv')
    parser.add_argument('--path_metadata', type=str)   
    parser.add_argument('--status_col', type=str) 
    parser.add_argument('--event_col', type=str, default='delosrandopointy') 
    parser.add_argument('--study', type=str, default='os')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
  
    args = parse_args()
    print(vars(args))
    
    # Print the dictionary

    df_clinical= pd.read_csv(args.path_clinical_csv,sep=';')
    df_clinical = df_clinical.loc[~df_clinical.dup_num_peace1]
    df_clinical["AGE_cat"] = pd.cut(x=df_clinical["AGE"].astype(float), bins=[0,59,69,float('inf')], labels=["Low","Medium","Large"])

    categorical = ['gleasonv','abirealv','ECOG','burdenv','AGE_cat','marqueurs_neuroendocrines', 'AR.Recode'] #'typecastv', 'docetaxelrealv'
   
    label_encoders = {}
    for feature in categorical:
        print(f'{feature}: {len(df_clinical[feature].unique())}')
        df_clinical[feature] = df_clinical[feature].astype(str)
        label_encoders[feature] = LabelEncoder()
        label_encoders[feature].fit(
            df_clinical.loc[:, feature])
        df_clinical.loc[ :,feature] = label_encoders[feature].transform(
                df_clinical.loc[:, feature])
    
    # remove marker slides
    df_marker_slides = pd.read_csv(args.path_marker_slides_csv,header=None)
    list_marker_slides = list(df_marker_slides[0].apply(lambda x: os.path.basename(x)[:-3]+'pt'))
    

    list_files = os.listdir(args.path_files)
    list_csv = []
    for in_file in list_files:
        in_file = os.path.join(args.path_files,in_file)
        if in_file in list_marker_slides:
            continue
        base_name = os.path.basename(in_file).split('.')[0]
        diamic_scan = base_name.split('_')[0]   
        dico_bin_labels = bin_labels[args.study] 

        if diamic_scan in list(df_clinical.diamic_scan):
            label_raw = df_clinical[df_clinical.diamic_scan == diamic_scan][args.status_col].item()
            event = df_clinical[df_clinical.diamic_scan == diamic_scan][args.event_col].item()
            status = dico_bin_labels[label_raw]
            
            case_id = df_clinical[df_clinical.diamic_scan == diamic_scan].SUBJID.item()
            gleasonv = df_clinical[df_clinical.diamic_scan == diamic_scan].gleasonv.item()
            ECOG = df_clinical[df_clinical.diamic_scan == diamic_scan].ECOG.item()
            burdenv = df_clinical[df_clinical.diamic_scan == diamic_scan].burdenv.item()
            AGE_cat = df_clinical[df_clinical.diamic_scan == diamic_scan].AGE_cat.item()
            abirealv=df_clinical[df_clinical.diamic_scan == diamic_scan].abirealv.item() #,'ECOG','burdenv','AGE_cat','typecastv', 'docetaxelrealv'
            NPEC=df_clinical[df_clinical.diamic_scan == diamic_scan].marqueurs_neuroendocrines.item()
            AR=df_clinical[df_clinical.diamic_scan == diamic_scan]['AR.Recode'].item()
            row_file =  [args.study, case_id,event,status,in_file,gleasonv,ECOG,burdenv,AGE_cat,NPEC,AR,abirealv] #Study,ID,Event,Status,WSI
            list_csv.append(row_file)
        else:
            continue 


    csv_data = pd.DataFrame(list_csv,columns=['Study','ID','Event','Status','WSI','gleasonv','ECOG','burdenv','AGE_cat','NPEC','AR','abirealv'])
    csv_data.to_csv(args.path_metadata, index=False)

