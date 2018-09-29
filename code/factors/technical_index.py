

from jaqs_fxdayu.data import DataView
from jaqs_fxdayu.data import RemoteDataService
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")


dv = DataView()
dataview_folder = 'F:\\DeepLearning\\python\\JAQS_Data\\000300.SH\\lb.daily'
dv.load_dataview(dataview_folder)






# In[8]:


name = "EMA10"
dv.add_formula(name, "Ta('EMA',0,open_adj,high_adj,low_adj,close_adj,volume,10)", is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[9]:


name = "EMA12"
dv.add_formula(name, "Ta('EMA',0,open_adj,high_adj,low_adj,close_adj,volume,12)", is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[10]:


name = "EMA120"
dv.add_formula(name, "Ta('EMA',0,open_adj,high_adj,low_adj,close_adj,volume,120)", is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[ ]:


name = "EMA20"
dv.add_formula(name, "Ta('EMA',0,open_adj,high_adj,low_adj,close_adj,volume,20)", is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[ ]:


name = "EMA26"
dv.add_formula(name, "Ta('EMA',0,open_adj,high_adj,low_adj,close_adj,volume,26)", is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[ ]:


name = "EMA5"
dv.add_formula(name, "Ta('EMA',0,open_adj,high_adj,low_adj,close_adj,volume,5)", is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[ ]:


name = "EMA60"
dv.add_formula(name, "Ta('EMA',0,open_adj,high_adj,low_adj,close_adj,volume,60)", is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[ ]:


name = "MA120"
dv.add_formula(name, "Ta('MA',0,open_adj,high_adj,low_adj,close_adj,volume,120)", is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[ ]:


name = "MA20"
dv.add_formula(name, "Ta('MA',0,open_adj,high_adj,low_adj,close_adj,volume,20)", is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[ ]:


name = "MA10"
dv.add_formula(name, "Ta('MA',0,open_adj,high_adj,low_adj,close_adj,volume,10)", is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[ ]:


name = "MA5"
dv.add_formula(name, "Ta('MA',0,open_adj,high_adj,low_adj,close_adj,volume,5)", is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[ ]:


name = "MA60"
dv.add_formula(name, "Ta('MA',0,open_adj,high_adj,low_adj,close_adj,volume,60)", is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[ ]:


name = "BBI"
dv.add_formula(name, "(Ta('MA',0,open_adj,high_adj,low_adj,close_adj,volume,3)+Ta('MA',0,open_adj,high_adj,low_adj,close_adj,volume,6)"
               +"+Ta('MA',0,open_adj,high_adj,low_adj,close_adj,volume,12)+Ta('MA',0,open_adj,high_adj,low_adj,close_adj,volume,24))/4", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[34]:


name = "BBIC"
dv.add_formula(name, "BBI/close_adj", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[35]:


name = "TEMA10"
dv.add_formula(name, "Ta('TEMA',0,open_adj,high_adj,low_adj,close_adj,volume,10)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[36]:


name = "TEMA5"
dv.add_formula(name, "Ta('TEMA',0,open_adj,high_adj,low_adj,close_adj,volume,5)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[37]:


name = "BollUp"
dv.add_formula(name, "Ta('BBANDS',0,open_adj,high_adj,low_adj,close_adj,volume,5)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[38]:


name = "BollMiddle"
dv.add_formula(name, "Ta('BBANDS',1,open_adj,high_adj,low_adj,close_adj,volume,5)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[39]:


name = "BollDown"
dv.add_formula(name, "Ta('BBANDS',2,open_adj,high_adj,low_adj,close_adj,volume,5)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[40]:


name = "slowK"
dv.add_formula(name, "Ta('STOCH',0,open_adj,high_adj,low_adj,close_adj,volume)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[41]:


name = "slowD"
dv.add_formula(name, "Ta('STOCH',1,open_adj,high_adj,low_adj,close_adj,volume)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[42]:


name = "slowJ"
dv.add_formula(name, "3 * slowK - 2 * slowD", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[43]:


name = "fastK"
dv.add_formula(name, "Ta('STOCHF',0,open_adj,high_adj,low_adj,close_adj,volume)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[44]:


name = "fastD"
dv.add_formula(name, "Ta('STOCHF',1,open_adj,high_adj,low_adj,close_adj,volume)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[45]:


name = "fastJ"
dv.add_formula(name, "3 * fastK - 2 * fastD", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[46]:


name = "fastK_RSI"
dv.add_formula(name, "Ta('STOCHRSI',0,open_adj,high_adj,low_adj,close_adj,volume)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[47]:


name = "fastD_RSI"
dv.add_formula(name, "Ta('STOCHRSI',1,open_adj,high_adj,low_adj,close_adj,volume)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[48]:


name = "fastJ_RSI"
dv.add_formula(name, "3 * fastK_RSI - 2 * fastD_RSI", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[49]:


name = "UpRVI"
dv.add_formula(name, "Ta('EMA',0,open_adj,high_adj,low_adj,StdDev(close_adj,10)*(Delta(close_adj,1)>0)+0,volume,2*10-1)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[50]:


name = "DownRVI"
dv.add_formula(name, "Ta('EMA',0,open_adj,high_adj,low_adj,StdDev(close_adj,10)*(Delta(close_adj,1)<0)+0,volume,2*10-1)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[51]:


name = "RVI"
dv.add_formula(name, "100 * UpRVI / (UpRVI + DownRVI)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[52]:


name = "DBCD"
dv.add_formula("DIF", "(close_adj/MA5-1)*100-Delay((close_adj/MA5-1)*100,16)", 
               is_quarterly=False, add_data=True)
dv.add_formula(name, "Ta('EMA',0,DIF,DIF,DIF,DIF,DIF,17)", 
               is_quarterly=False, add_data=True)
dv.remove_field("DIF")
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[53]:


name = "MFI"
dv.add_formula(name, "Ta('MFI',0,open_adj,high_adj,low_adj,close_adj,volume,14)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[54]:


name = "CR20"
dv.add_formula("TYP", "(high_adj+low_adj+close_adj)/3", 
               is_quarterly=False, add_data=True)
dv.add_formula(name, "Ts_Sum(Max(high_adj-TYP,0),20)/Ts_Sum(Max(TYP-low_adj,0),20)*100", 
               is_quarterly=False, add_data=True)
dv.remove_field("TYP")
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[55]:


name = "MassIndex"
dv.add_formula("EMAHL", "Ta('EMA',0,0,0,0,high_adj-low_adj,9)", 
               is_quarterly=False, add_data=True)
dv.add_formula("EMARatio", "EMAHL/Ta('EMA',0,0,0,0,EMAHL,9)", 
               is_quarterly=False, add_data=True)
dv.add_formula(name, "Ts_Sum(EMARatio,25)", 
               is_quarterly=False, add_data=True)
dv.remove_field("EMAHL")
dv.remove_field("EMARatio")
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[56]:


name = "Elder"
dv.add_formula("BullPower", "high_adj - Ta('EMA',0,0,0,0,close_adj,13)", 
               is_quarterly=False, add_data=True)
dv.add_formula("BearPower", "low_adj - Ta('EMA',0,0,0,0,close_adj,13)", 
               is_quarterly=False, add_data=True)
dv.add_formula(name, "(BullPower-BearPower)/close_adj", 
               is_quarterly=False, add_data=True)
dv.remove_field("BullPower")
dv.remove_field("BearPower")
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[58]:


name = "CLV"
dv.add_formula(name, "(2*close_adj-high_adj-low_adj)/(high_adj-low_adj)*volume / 1000000", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[59]:


name = "ChaikinVolatility"
dv.add_formula("HLEMA", "Ta('EMA',0,0,0,0,high_adj-low_adj,10)", 
               is_quarterly=False, add_data=True)
dv.add_formula(name, "100*Delta(HLEMA,10)/Delay(HLEMA,10)", 
               is_quarterly=False, add_data=True)
dv.remove_field("HLEMA")
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[60]:


name = "EMV14"
dv.add_formula(name, "Ta('EMA',0,0,0,0,((high_adj+low_adj)/2-(Delay(high_adj,1)+Delay(low_adj,1))/2)*(high_adj-low_adj)/volume,14)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[61]:


name = "EMV6"
dv.add_formula(name, "Ta('EMA',0,0,0,0,((high_adj+low_adj)/2-(Delay(high_adj,1)+Delay(low_adj,1))/2)*(high_adj-low_adj)/volume,6)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[62]:


name = "MINUS_DI"
dv.add_formula(name, "Ta('MINUS_DI',0,open_adj,high_adj,low_adj,close_adj,volume,14)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[63]:


name = "MINUS_DM"
dv.add_formula(name, "Ta('MINUS_DM',0,open_adj,high_adj,low_adj,close_adj,volume,14)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[64]:


name = "PLUS_DI"
dv.add_formula(name, "Ta('PLUS_DI',0,open_adj,high_adj,low_adj,close_adj,volume,14)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[65]:


name = "PLUS_DM"
dv.add_formula(name, "Ta('PLUS_DM',0,open_adj,high_adj,low_adj,close_adj,volume,14)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[66]:


name = "DEMA"
dv.add_formula(name, "Ta('DEMA',0,open_adj,high_adj,low_adj,close_adj,volume,30)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[67]:


name = "KAMA"
dv.add_formula(name, "Ta('KAMA',0,open_adj,high_adj,low_adj,close_adj,volume,30)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[68]:


name = "MAMA"
dv.add_formula(name, "Ta('MAMA',0,open_adj,high_adj,low_adj,close_adj,volume,0.5,0.05)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[69]:


name = "FAMA"
dv.add_formula(name, "Ta('MAMA',1,open_adj,high_adj,low_adj,close_adj,volume,0.5,0.05)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[70]:


name = "MIDPOINT"
dv.add_formula(name, "Ta('MIDPOINT',0,open_adj,high_adj,low_adj,close_adj,volume,14)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[71]:


name = "MIDPRICE"
dv.add_formula(name, "Ta('MIDPRICE',0,open_adj,high_adj,low_adj,close_adj,volume,14)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[72]:


name = "SAR"
dv.add_formula(name, "Ta('SAR',0,open_adj,high_adj,low_adj,close_adj,volume,0.02,0.2)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[73]:


name = "SAREXT"
dv.add_formula(name, "Ta('SAR',0,open_adj,high_adj,low_adj,close_adj,volume,0.02,0.2)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[74]:


name = "T3"
dv.add_formula(name, "Ta('T3',0,open_adj,high_adj,low_adj,close_adj,volume,5,0.7)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[75]:


name = "TRIMA"
dv.add_formula(name, "Ta('TRIMA',0,open_adj,high_adj,low_adj,close_adj,volume,30)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[76]:


name = "ADX"
dv.add_formula(name, "Ta('ADX',0,open_adj,high_adj,low_adj,close_adj,volume,14)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[77]:


name = "ADXR"
dv.add_formula(name, "Ta('ADXR',0,open_adj,high_adj,low_adj,close_adj,volume,14)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[78]:


name = "APO"
dv.add_formula(name, "Ta('APO',0,open_adj,high_adj,low_adj,close_adj,volume,12,26)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[82]:


name = "AROONdown"
dv.add_formula(name, "Ta('AROON',0,open_adj,high_adj,low_adj,close_adj,volume,14)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[83]:


name = "AROONup"
dv.add_formula(name, "Ta('AROON',1,open_adj,high_adj,low_adj,close_adj,volume,14)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[84]:


name = "AROON"
dv.add_formula(name, "AROONdown-AROONup", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[86]:


name = "AROONOSC"
dv.add_formula(name, "Ta('AROONOSC',0,open_adj,high_adj,low_adj,close_adj,volume,14)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[87]:


name = "BOP"
dv.add_formula(name, "Ta('BOP',0,open_adj,high_adj,low_adj,close_adj,volume)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[88]:


name = "CCI"
dv.add_formula(name, "Ta('CCI',0,open_adj,high_adj,low_adj,close_adj,volume,14)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[89]:


name = "CMO"
dv.add_formula(name, "Ta('CMO',0,open_adj,high_adj,low_adj,close_adj,volume,14)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[90]:


name = "DX"
dv.add_formula(name, "Ta('DX',0,open_adj,high_adj,low_adj,close_adj,volume,14)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[91]:


name = "MACD"
dv.add_formula(name, "Ta('MACD',0,open_adj,high_adj,low_adj,close_adj,volume,12,26,9)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[92]:


name = "MACDsignal"
dv.add_formula(name, "Ta('MACD',1,open_adj,high_adj,low_adj,close_adj,volume,12,26,9)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[93]:


name = "MACDhist"
dv.add_formula(name, "Ta('MACD',2,open_adj,high_adj,low_adj,close_adj,volume,12,26,9)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[95]:


name = "MOM"
dv.add_formula(name, "Ta('MOM',0,open_adj,high_adj,low_adj,close_adj,volume,10)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[96]:


name = "PPO"
dv.add_formula(name, "Ta('PPO',0,open_adj,high_adj,low_adj,close_adj,volume,12,26)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[97]:


name = "ROC"
dv.add_formula(name, "Ta('ROC',0,open_adj,high_adj,low_adj,close_adj,volume,10)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[98]:


name = "ROCP"
dv.add_formula(name, "Ta('ROCP',0,open_adj,high_adj,low_adj,close_adj,volume,10)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[99]:


name = "ROCR"
dv.add_formula(name, "Ta('ROCR',0,open_adj,high_adj,low_adj,close_adj,volume,10)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[100]:


name = "ROCR100"
dv.add_formula(name, "Ta('ROCR100',0,open_adj,high_adj,low_adj,close_adj,volume,10)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[101]:


name = "RSI"
dv.add_formula(name, "Ta('RSI',0,open_adj,high_adj,low_adj,close_adj,volume,14)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[102]:


name = "TRIX"
dv.add_formula(name, "Ta('TRIX',0,open_adj,high_adj,low_adj,close_adj,volume,30)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[103]:


name = "ULTOSC"
dv.add_formula(name, "Ta('ULTOSC',0,open_adj,high_adj,low_adj,close_adj,volume,7,14,28)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[104]:


name = "WILLR"
dv.add_formula(name, "Ta('WILLR',0,open_adj,high_adj,low_adj,close_adj,volume,14)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[105]:


name = "AD"
dv.add_formula(name, "Ta('AD',0,open_adj,high_adj,low_adj,close_adj,volume)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[106]:


name = "ADOSC"
dv.add_formula(name, "Ta('ADOSC',0,open_adj,high_adj,low_adj,close_adj,volume,3,10)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[107]:


name = "OBV"
dv.add_formula(name, "Ta('OBV',0,open_adj,high_adj,low_adj,close_adj,volume)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[108]:


name = "ATR"
dv.add_formula(name, "Ta('ATR',0,open_adj,high_adj,low_adj,close_adj,volume,14)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[109]:


name = "NATR"
dv.add_formula(name, "Ta('NATR',0,open_adj,high_adj,low_adj,close_adj,volume,14)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[110]:


name = "TRANGE"
dv.add_formula(name, "Ta('TRANGE',0,open_adj,high_adj,low_adj,close_adj,volume,10)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[111]:


name = "AVGPRICE"
dv.add_formula(name, "Ta('AVGPRICE',0,open_adj,high_adj,low_adj,close_adj,volume)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[112]:


name = "MEDPRICE"
dv.add_formula(name, "Ta('MEDPRICE',0,open_adj,high_adj,low_adj,close_adj,volume)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[113]:


name = "TYPPRICE"
dv.add_formula(name, "Ta('TYPPRICE',0,open_adj,high_adj,low_adj,close_adj,volume)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[114]:


name = "WCLPRICE"
dv.add_formula(name, "Ta('WCLPRICE',0,open_adj,high_adj,low_adj,close_adj,volume)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[115]:


name = "HT_DCPERIOD"
dv.add_formula(name, "Ta('HT_DCPERIOD',0,open_adj,high_adj,low_adj,close_adj,volume)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)
dv.get_ts(name).tail()


# In[116]:


name = "HT_DCPHASE"
dv.add_formula(name, "Ta('HT_DCPHASE',0,open_adj,high_adj,low_adj,close_adj,volume)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[117]:


name = "INPHASE"
dv.add_formula(name, "Ta('HT_PHASOR',0,open_adj,high_adj,low_adj,close_adj,volume)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[121]:


name = "QUADRATURE"
dv.add_formula(name, "Ta('HT_PHASOR',1,open_adj,high_adj,low_adj,close_adj,volume)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[119]:


name = "SINE"
dv.add_formula(name, "Ta('HT_SINE',0,open_adj,high_adj,low_adj,close_adj,volume)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[122]:


name = "LEADSINE"
dv.add_formula(name, "Ta('HT_SINE',1,open_adj,high_adj,low_adj,close_adj,volume)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[123]:


name = "INTEGER"
dv.add_formula(name, "Ta('HT_TRENDMODE',0,open_adj,high_adj,low_adj,close_adj,volume)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[125]:


name = "TR"
dv.add_formula(name, "Max(high_adj,Delay(close_adj,1))-Min(low_adj,Delay(close_adj,1))", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[128]:


name = "XR"
dv.add_formula(name, "close_adj-Min(low_adj,Delay(close_adj,1))", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[129]:


name = "XRM"
dv.add_formula(name, "Ts_Sum(XR,7)/Ts_Sum(TR,7)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[130]:


name = "XRN"
dv.add_formula(name, "Ts_Sum(XR,14)/Ts_Sum(TR,14)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[131]:


name = "XRO"
dv.add_formula(name, "Ts_Sum(XR,28)/Ts_Sum(TR,28)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[132]:


name = "UOS"
dv.add_formula(name, "100 * (XRO*7*14+XRN*7*28+XRM*14*28) / (7*14+7*28+14*28)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[141]:


name = "SRMI"
dv.add_formula(name, "Delta(close_adj,10)/Max(close_adj,Delay(close_adj,10))", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[142]:


name = "REV5M20"
dv.add_formula(name, "close_adj/Delay(close_adj,5)-close_adj/Delay(close_adj,20)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[147]:


name = "COPPOCK"
dv.add_formula(name, "Ta('WMA',0,0,0,0,100*(close_adj/Delay(close_adj,14)+close_adj/Delay(close_adj,11)),0,10)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[146]:


name = "DEA"
dv.add_formula(name, "Ta('EMA',0,0,0,0,Ta('EMA',0,0,0,0,close_adj,0,12)-Ta('EMA',0,0,0,0,close_adj,0,26),0,9)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[152]:


name = "DDI"
dv.add_formula("DMZ", "Max(Abs(Delta(high_adj,1)),Abs(Delta(close_adj,1)))*(0+(high_adj+low_adj)>(Delay(high_adj,1)+Delay(low_adj,1)))", 
               is_quarterly=False, add_data=True)
dv.add_formula("DMF", "Max(Abs(Delta(high_adj,1)),Abs(Delta(close_adj,1)))*(0+(high_adj+low_adj)<(Delay(high_adj,1)+Delay(low_adj,1)))", 
               is_quarterly=False, add_data=True)
dv.add_formula("DIZ", "Ts_Sum(DMZ,13)/(Ts_Sum(DMZ,13)+Ts_Sum(DMF,13))", 
               is_quarterly=False, add_data=True)
dv.add_formula("DIF", "Ts_Sum(DMF,13)/(Ts_Sum(DMZ,13)+Ts_Sum(DMF,13))", 
               is_quarterly=False, add_data=True)
dv.add_formula(name, "DIZ-DIF", 
               is_quarterly=False, add_data=True)
dv.remove_field("DMZ")
dv.remove_field("DMF")
dv.remove_field("DIZ")
dv.remove_field("DIF")
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[155]:


name = "ARBR"
dv.add_formula("AR", "Ts_Sum(high_adj-open_adj,26)/Ts_Sum(open_adj-low_adj,26)", 
               is_quarterly=False, add_data=True)
dv.add_formula("BR", "Ts_Sum(high_adj-close_adj,26)/Ts_Sum(close_adj-low_adj,26)", 
               is_quarterly=False, add_data=True)
dv.add_formula(name, "AR-BR", 
               is_quarterly=False, add_data=True)
dv.remove_field("AR")
dv.remove_field("BR")
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[158]:


name = "PSY"
dv.add_formula(name, "Ts_Sum(Delta(close_adj,1)>0 + 0,12) / 12", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[162]:


name = "WVAD"
dv.add_formula("WVAD", "Ts_Sum((close_adj-open_adj)/(high_adj-low_adj)*volume,24)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[164]:


name = "MAWVAD"
dv.add_formula("MAWVAD", "Ta('SMA',0,0,0,0,WVAD,0,6)", 
               is_quarterly=False, add_data=True)
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[163]:


name = "ADTM"
dv.add_formula("DTM", "Max(high_adj-open_adj,Delta(open_adj,1))*(0+Delta(open_adj,1)>0)", 
               is_quarterly=False, add_data=True)
dv.add_formula("DBM", "Max(high_adj-open_adj,Delta(open_adj,1))*(0+Delta(open_adj,1)<0)", 
               is_quarterly=False, add_data=True)
dv.add_formula("STM", "Ts_Sum(DTM,20)", 
               is_quarterly=False, add_data=True)
dv.add_formula("SBM", "Ts_Sum(DBM,20)", 
               is_quarterly=False, add_data=True)
dv.add_formula(name, "(STM-SBM)/Max(STM,SBM)", 
               is_quarterly=False, add_data=True)
dv.remove_field("DTM")
dv.remove_field("DBM")
dv.remove_field("STM")
dv.remove_field("SBM")
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[165]:


name = "ACD6"
dv.add_formula("buy", "close_adj-Min(low_adj,Delay(close_adj,1))", 
               is_quarterly=False, add_data=True)
dv.add_formula("sell", "close_adj-Max(high_adj,Delay(close_adj,1))", 
               is_quarterly=False, add_data=True)
dv.add_formula(name, "Ts_Sum(buy,6)+Ts_Sum(sell,6)", 
               is_quarterly=False, add_data=True)
dv.remove_field("buy")
dv.remove_field("sell")
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[166]:


name = "ACD20"
dv.add_formula("buy", "close_adj-Min(low_adj,Delay(close_adj,1))", 
               is_quarterly=False, add_data=True)
dv.add_formula("sell", "close_adj-Max(high_adj,Delay(close_adj,1))", 
               is_quarterly=False, add_data=True)
dv.add_formula(name, "Ts_Sum(buy,20)+Ts_Sum(sell,20)", 
               is_quarterly=False, add_data=True)
dv.remove_field("buy")
dv.remove_field("sell")
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()
？？

# In[167]:


name = "VMACD"
dv.add_formula("VDIFF", "Ta('EMA',0,0,0,0,volume,0,12)-Ta('EMA',0,0,0,0,volume,0,26)", 
               is_quarterly=False, add_data=True)
dv.add_formula("VDEA", "Ta('EMA',0,0,0,0,VDIFF,0,9)", 
               is_quarterly=False, add_data=True)
dv.add_formula(name, "VDIFF-VDEA", 
               is_quarterly=False, add_data=True)
dv.remove_field("VDIFF")
dv.remove_field("VDEA")
print(dv.get_ts(name).shape)

dv.get_ts(name).tail()


# In[180]:


dv.save_dataview(dataview_folder)


