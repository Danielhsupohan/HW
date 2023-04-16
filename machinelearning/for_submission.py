import joblib 
model_pretrained=joblib.load('sales_lr.pkl')
import pandas as pd

df_test=pd.read_csv("test.csv")
df_test
df_test.drop(['id'],axis=1) 
df_test.head()

predictions2=model_pretrained.predict(df_test.drop(['id'],axis=1) )

predictions2

forSubmissionDF=pd.DataFrame(columns=['id','target'])
forSubmissionDF
forSubmissionDF['id']=range(414,690)
forSubmissionDF['target']=predictions2
forSubmissionDF

forSubmissionDF.to_csv('s.csv',index=False)
