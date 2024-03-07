import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer

from dataclasses import dataclass

@dataclass
class DataPreProcessingFEConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")

class DataPreProcessingFE:
    def __init__(self):
        self.preprocessing_config=DataPreProcessingFEConfig()

    def initiate_data_preprocessing(self):
            logging.info("Entered the data preprocessing method or component")
            try:
                data=pd.read_csv('artifacts/data.csv')
                logging.info('Read the raw dataset needed for pre processing')


                #modifying columns as per EDA file
                data.drop(columns='Unnamed: 0',inplace=True)

                logging.info('Datetime conversion started')

                ##Converting 'MMM-YY' feature to datetime type
                data["MMM-YY"] = pd.to_datetime(data["MMM-YY"])

                ##Converting 'Dateofjoining' feature to datetime type
                data['Dateofjoining'] = pd.to_datetime(data['Dateofjoining'])

                ##Converting 'LastWorkingDate' feature to datetime type
                data['LastWorkingDate'] = pd.to_datetime(data['LastWorkingDate'])

                logging.info('Datetime conversion ended')

                data_nums=data.select_dtypes(np.number)
                data_nums.drop(columns='Driver_ID',inplace=True)
                columns = data_nums.columns
                imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean',)
                imputer.fit(data_nums)
                # transform the dataset
                data_new = imputer.transform(data_nums)
                data_new=pd.DataFrame(data_new)
                data_new.columns=columns
                remaining_columns=list(set(data.columns).difference(set(columns)))
                data=pd.concat([data_new, data[remaining_columns]],axis=1)

                function_dict = {'Age':'max', 'Gender':'first','City':'first',
                 'Education_Level':'last', 'Income':'last', 
                 'Joining Designation':'last','Grade':'last', 
                 'Dateofjoining':'last','LastWorkingDate':'last',
                 'Total Business Value':'sum','Quarterly Rating':'last'}
                new_train=data.groupby(['Driver_ID','MMM-YY']).aggregate(function_dict)
                df=new_train.sort_index( ascending=[True,True])

                df1=pd.DataFrame()

                df1['Driver_ID']=data['Driver_ID'].unique()

                df1['Age'] = list(data.groupby('Driver_ID',axis=0).max('MMM-YY')['Age'])
                df1['Gender'] = list(data.groupby('Driver_ID').agg({'Gender':'last'})['Gender'])
                df1['City'] = list(data.groupby('Driver_ID').agg({'City':'last'})['City'])
                df1['Education'] = list(data.groupby('Driver_ID').agg({'Education_Level':'last'})['Education_Level'])
                df1['Income'] = list(data.groupby('Driver_ID').agg({'Income':'last'})['Income'])
                df1['Joining_Designation'] = list(data.groupby('Driver_ID').agg({'Joining Designation':'last'})['Joining Designation'])
                df1['Grade'] = list(data.groupby('Driver_ID').agg({'Grade':'last'})['Grade'])
                df1['Total_Business_Value'] = list(data.groupby('Driver_ID',axis=0).sum('Total Business Value')['Total Business Value'])
                df1['Last_Quarterly_Rating'] = list(data.groupby('Driver_ID').agg({'Quarterly Rating':'last'})['Quarterly Rating'])

                #Quarterly rating at the beginning
                qrf = df.groupby('Driver_ID').agg({'Quarterly Rating':'first'})

                #Quarterly rating at the end
                qrl = df.groupby('Driver_ID').agg({'Quarterly Rating':'last'})

                #The dataset which has the employee ids and a bollean value which tells if the rating has increased
                qr = (qrl['Quarterly Rating']>qrf['Quarterly Rating']).reset_index()

                #the employee ids whose rating has increased
                empid = qr[qr['Quarterly Rating']==True]['Driver_ID']

                qri = []
                for i in df1['Driver_ID']:
                    if i in empid.values:  # changed -- instead of empid--> empid.values
                        qri.append(1)
                    else:
                        qri.append(0)

                df1['Quarterly_Rating_Increased'] = qri

                lwr = (df.groupby('Driver_ID').agg({'LastWorkingDate':'last'})['LastWorkingDate'].isna()).reset_index()
                #The employee ids who do not have last working date
                empid = lwr[lwr['LastWorkingDate']==True]['Driver_ID']

                target = []
                for i in df1['Driver_ID']:
                    if i in empid.values:
                        target.append(0)
                    elif i not in empid.values:
                        target.append(1)
                        
                df1['Target'] = target

                #Quarterly rating at the beginning
                sf = df.groupby('Driver_ID').agg({'Income':'first'})

                #Quarterly rating at the end
                sl = df.groupby('Driver_ID').agg({'Income':'last'})

                #The dataset which has the employee ids and a bollean value which tells if the monthly income has increased
                s = (sl['Income']>sf['Income']).reset_index()

                #the employee ids whose monthly income has increased
                empid = s[s['Income']==True]['Driver_ID']

                si = []
                for i in df1['Driver_ID']:
                    if i in empid.values:
                        si.append(1)
                    else:
                        si.append(0)
                df1['Income_Increased'] = si

                print(df1.head())

            except:
                 pass
            
if __name__=="__main__":
    logging.info("test datappfe")
    obj = DataPreProcessingFE()
    obj.initiate_data_preprocessing()
