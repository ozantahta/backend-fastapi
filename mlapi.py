from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd 
import pickle
import pymongo 
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
warnings.simplefilter('ignore')

myclient = pymongo.MongoClient('mongodb+srv://username:password@cluster0.ncjbb.mongodb.net/?retryWrites=true&w=majority')
mydb = myclient['churn_data']
mycol = mydb['daily_data']
myrealcol = mydb['real_daily_data']

app = FastAPI()

origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScoringItem(BaseModel):
    duration_mean: float
    duration_std:  float
    duration_cv:  float
    duration_total_time_spent: float
    sum_revenue: float
    no_rewarded_impression: float
    std_coin: float
    mean_coin:  float
    last_coin:  float
    x0: float
    x1: float
    x2: float
    x3: float
    x4: float
    x5: float
    x6: float
    x7: float
    x8: float
    x9: float
    x10: float
    x11: float
    x12: float
    x13: float
    x14: float
    x15: float
    x16: float
    x17: float
    x18: float
    x19: float
    x20: float
    x21: float
    x22: float
    x23: float
    x24: float
    x25: float
    x26: float
    x27: float
    x28: float
    x29: float
    x30: float
    x31: float
    x32: float
    x33: float
    x34: float
    x35: float
    x36: float
    x37: float
    x38: float
    x39: float
    x40: float
    x41: float
    x42: float
    x43: float
    x44: float
    x45: float
    x46: float
    x47: float
    x48: float
    x49: float
    x50: float
    x51: float
    x52: float
    x53: float
    x54: float
    x55: float
    x56: float
    x57: float
    x58: float
    x59: float
    x60: float
    x61: float
    x62: float
    x63: float
    x64: float
    x65: float
    x66: float
    x67: float
    x68: float
    x69: float
    x70: float
    x71: float
    x72: float
    x73: float
    x74: float
    x75: float
    x76: float
    x77: float
    x78: float
    x79: float
    x80: float
    x81: float
    x82: float
    x83: float
    x84: float
    x85: float
    x86: float
    x87: float
    x88: float
    x89: float
    x90: float
    x91: float
    x92: float
    x93: float
    x94: float
    x95: float
    x96: float
    x97: float
    x98: float
    x99: float
    x100: float
    x101: float
    x102: float
    x103: float
    x104: float
    x105: float
    x106: float
    x107: float
    x108: float
    x109: float
    x110: float
    x111: float
    x112: float
    x113: float
    x114: float
    x115: float
    x116: float
    x117: float
    x118: float
    x119: float
    x120: float
    x121: float
    x122: float
    x123: float
    x124: float
    x125: float
    x126: float
    x127: float
    x128: float
    x129: float

with open('logistic_reg_model.sav', 'rb') as f:
    model = pickle.load(f)

@app.get('/check')
async def info_check():
    return {'check': "Hello, It works very well!! ✔️"}

@app.get('/getData')
async def getData():
    all_dict = []
    for hour_time in myrealcol.find({}):
        hour_time.pop('_id')
        hour_time.pop('churned')
        all_dict.append(hour_time)
    new_dict = sorted(all_dict, key=lambda x: x['count'])
    return new_dict

@app.get('/getPredictions')
async def getPred():
    all_dict = []
    for hour_time in mycol.find({}):
        hour_time.pop('_id')
        all_dict.append(hour_time)
    new_dict = sorted(all_dict, key=lambda x: x['count'])
    df = pd.DataFrame(new_dict)
    df1 = df.drop(['count', 'time'], axis=1).to_numpy()
    y_head = model.predict(df1)
    temp_df = pd.DataFrame({'time':df['time'].tolist(), 'churned': y_head.tolist()})
    temp_dict = temp_df.to_dict('records')
    # res = {df['time'].tolist()[i]: y_head.tolist()[i] for i in range(len(y_head))}
    return temp_dict

@app.post('/')
async def scoring_endpoint(item: ScoringItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    y_head = int(model.predict(df))
    # final_dict = item.dict()
    # final_dict.update({'churned': y_head})
    # return {'churned': y_head}
    return item
    

