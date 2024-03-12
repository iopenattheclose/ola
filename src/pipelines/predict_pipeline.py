import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
             Age: int,
             Gender: int,
             Education: int,
             Income: int,
             Joining_Designation: int,
             Grade: int,
             Total_Business_Value: int,
             Last_Quarterly_Rating: int,
             Quarterly_Rating_Increased: int,
             Income_Increased: int,
             City_C1:  int,
             City_C10: int,
             City_C11: int,
             City_C12: int,
             City_C13: int,
             City_C14: int,
             City_C15: int,
             City_C16: int,
             City_C17: int,
             City_C18: int,
             City_C19: int,
             City_C2:  int,
             City_C20: int,
             City_C21: int,
             City_C22: int,
             City_C23: int,
             City_C24: int,
             City_C25: int,
             City_C26: int,
             City_C27: int,
             City_C28: int,
             City_C29: int,
             City_C3: int,
             City_C4: int,
             City_C5: int,
             City_C6: int,
             City_C7: int,
             City_C8: int,
             City_C9: int
             ):
        self.Age = Age
        self.Gender = Gender
        self.Education = Education
        self.Income = Income
        self.Joining_Designation = Joining_Designation
        self.Grade = Grade
        self.Total_Business_Value = Total_Business_Value
        self.Last_Quarterly_Rating = Last_Quarterly_Rating
        self.Quarterly_Rating_Increased = Quarterly_Rating_Increased
        self.Income_Increased = Income_Increased
        self.City_C1 = City_C1
        self.City_C10 = City_C10
        self.City_C11 = City_C11
        self.City_C12 = City_C12
        self.City_C13 = City_C13
        self.City_C14 = City_C14
        self.City_C15 = City_C15
        self.City_C16 = City_C16
        self.City_C17 = City_C17
        self.City_C18 = City_C18
        self.City_C19 = City_C19
        self.City_C2 = City_C2
        self.City_C20 = City_C20
        self.City_C21 = City_C21
        self.City_C22 = City_C22
        self.City_C23 = City_C23
        self.City_C24 = City_C24
        self.City_C25 = City_C25
        self.City_C26 = City_C26
        self.City_C27 = City_C27
        self.City_C28 = City_C28
        self.City_C29 = City_C29
        self.City_C3 = City_C3
        self.City_C4 = City_C4
        self.City_C5 = City_C5
        self.City_C6 = City_C6
        self.City_C7 = City_C7
        self.City_C8 = City_C8
        self.City_C9 = City_C9


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "Gender": [self.Gender],
                "Education": [self.Education],
                "Income": [self.Income],
                "Joining_Designation": [self.Joining_Designation],
                "Grade": [self.Grade],
                "Total_Business_Value": [self.Total_Business_Value],
                "Last_Quarterly_Rating": [self.Last_Quarterly_Rating],
                "Quarterly_Rating_Increased": [self.Quarterly_Rating_Increased],
                "Income_Increased": [self.Income_Increased],
                "City_C1": [self.City_C1],
                "City_C10": [self.City_C10],
                "City_C11": [self.City_C11],
                "City_C12": [self.City_C12],
                "City_C13": [self.City_C13],
                "City_C14": [self.City_C14],
                "City_C15": [self.City_C15],
                "City_C16": [self.City_C16],
                "City_C17": [self.City_C17],
                "City_C18": [self.City_C18],
                "City_C19": [self.City_C19],
                "City_C2": [self.City_C2],
                "City_C20": [self.City_C20],
                "City_C21": [self.City_C21],
                "City_C22": [self.City_C22],
                "City_C23": [self.City_C23],
                "City_C24": [self.City_C24],
                "City_C25": [self.City_C25],
                "City_C26": [self.City_C26],
                "City_C27": [self.City_C27],
                "City_C28": [self.City_C28],
                "City_C29": [self.City_C29],
                "City_C3": [self.City_C3],
                "City_C4": [self.City_C4],
                "City_C5": [self.City_C5],
                "City_C6": [self.City_C6],
                "City_C7": [self.City_C7],
                "City_C8": [self.City_C8],
                "City_C9": [self.City_C9]
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)