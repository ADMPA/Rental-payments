

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_summary import DataFrameSummary as DFS
import math
from sklearn.metrics import mean_squared_error
from datetime import datetime
from datetime import timedelta

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.preprocessing import StandardScaler, MinMaxScaler


```

# Data loading and initial description


```python
df = pd.read_csv('./data.csv')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>name</th>
      <th>dob</th>
      <th>houseID</th>
      <th>houseZip</th>
      <th>paymentDate</th>
      <th>paymentAmount</th>
      <th>rentAmount</th>
      <th>Unnamed: 7</th>
      <th>age</th>
      <th>year</th>
      <th>late</th>
      <th>pay_more</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Karima Germany</td>
      <td>1951-05-23</td>
      <td>1192.0</td>
      <td>92154</td>
      <td>2011-11-01</td>
      <td>1321.0</td>
      <td>1321.0</td>
      <td>NaN</td>
      <td>67.0</td>
      <td>1951.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Agustina Spargo</td>
      <td>2000-01-01</td>
      <td>21.0</td>
      <td>92111</td>
      <td>2011-09-06</td>
      <td>2289.0</td>
      <td>2289.0</td>
      <td>NaN</td>
      <td>18.0</td>
      <td>2000.0</td>
      <td>6.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Lucilla Broderick</td>
      <td>2000-01-01</td>
      <td>1474.0</td>
      <td>92159</td>
      <td>2011-11-01</td>
      <td>1439.0</td>
      <td>1439.0</td>
      <td>NaN</td>
      <td>18.0</td>
      <td>2000.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Russ Mchale</td>
      <td>1977-04-20</td>
      <td>2015.0</td>
      <td>92137</td>
      <td>2012-07-01</td>
      <td>1744.0</td>
      <td>1744.0</td>
      <td>NaN</td>
      <td>41.0</td>
      <td>1977.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Carmelita Ritzer</td>
      <td>1969-03-09</td>
      <td>311.0</td>
      <td>92136</td>
      <td>2011-02-01</td>
      <td>1471.0</td>
      <td>1471.0</td>
      <td>NaN</td>
      <td>49.0</td>
      <td>1969.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop('Unnamed: 7', axis=1, inplace=True)
```


```python
pd.options.display.float_format = '{:.2f}'.format
```


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>houseID</th>
      <th>houseZip</th>
      <th>paymentAmount</th>
      <th>rentAmount</th>
      <th>age</th>
      <th>year</th>
      <th>late</th>
      <th>pay_more</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>59813.00</td>
      <td>59813.00</td>
      <td>59813.00</td>
      <td>59813.00</td>
      <td>59813.00</td>
      <td>59813.00</td>
      <td>59813.00</td>
      <td>59813.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1236.78</td>
      <td>92147.07</td>
      <td>1528.67</td>
      <td>1505.65</td>
      <td>33.55</td>
      <td>1988.90</td>
      <td>2.36</td>
      <td>10780.55</td>
    </tr>
    <tr>
      <th>std</th>
      <td>715.83</td>
      <td>29.54</td>
      <td>316.89</td>
      <td>303.69</td>
      <td>23.84</td>
      <td>26.96</td>
      <td>4.94</td>
      <td>464967.20</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00</td>
      <td>92101.00</td>
      <td>428.00</td>
      <td>428.00</td>
      <td>0.00</td>
      <td>1900.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>610.00</td>
      <td>92121.00</td>
      <td>1323.00</td>
      <td>1310.00</td>
      <td>18.00</td>
      <td>1979.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1240.00</td>
      <td>92145.00</td>
      <td>1524.00</td>
      <td>1506.00</td>
      <td>20.00</td>
      <td>2000.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1860.00</td>
      <td>92171.00</td>
      <td>1741.00</td>
      <td>1720.00</td>
      <td>43.00</td>
      <td>2000.00</td>
      <td>2.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2475.00</td>
      <td>92199.00</td>
      <td>2861.00</td>
      <td>2647.00</td>
      <td>118.00</td>
      <td>2067.00</td>
      <td>31.00</td>
      <td>20119632.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 59813 entries, 0 to 59812
    Data columns (total 11 columns):
    name             59813 non-null object
    dob              59813 non-null object
    houseID          59813 non-null float64
    houseZip         59813 non-null int64
    paymentDate      59813 non-null object
    paymentAmount    59813 non-null float64
    rentAmount       59813 non-null float64
    age              59813 non-null float64
    year             59813 non-null float64
    late             59813 non-null float64
    pay_more         59813 non-null float64
    dtypes: float64(7), int64(1), object(3)
    memory usage: 5.0+ MB



```python
df.shape
```




    (59813, 11)




```python
dfs = DFS(df)
dfs.summary()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>dob</th>
      <th>houseID</th>
      <th>houseZip</th>
      <th>paymentDate</th>
      <th>paymentAmount</th>
      <th>rentAmount</th>
      <th>age</th>
      <th>year</th>
      <th>late</th>
      <th>pay_more</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>59813.00</td>
      <td>59813.00</td>
      <td>NaN</td>
      <td>59813.00</td>
      <td>59813.00</td>
      <td>59813.00</td>
      <td>59813.00</td>
      <td>59813.00</td>
      <td>59813.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1236.78</td>
      <td>92147.07</td>
      <td>NaN</td>
      <td>1528.67</td>
      <td>1505.65</td>
      <td>33.55</td>
      <td>1988.90</td>
      <td>2.36</td>
      <td>10780.55</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>715.83</td>
      <td>29.54</td>
      <td>NaN</td>
      <td>316.89</td>
      <td>303.69</td>
      <td>23.84</td>
      <td>26.96</td>
      <td>4.94</td>
      <td>464967.20</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.00</td>
      <td>92101.00</td>
      <td>NaN</td>
      <td>428.00</td>
      <td>428.00</td>
      <td>0.00</td>
      <td>1900.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>610.00</td>
      <td>92121.00</td>
      <td>NaN</td>
      <td>1323.00</td>
      <td>1310.00</td>
      <td>18.00</td>
      <td>1979.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1240.00</td>
      <td>92145.00</td>
      <td>NaN</td>
      <td>1524.00</td>
      <td>1506.00</td>
      <td>20.00</td>
      <td>2000.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1860.00</td>
      <td>92171.00</td>
      <td>NaN</td>
      <td>1741.00</td>
      <td>1720.00</td>
      <td>43.00</td>
      <td>2000.00</td>
      <td>2.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2475.00</td>
      <td>92199.00</td>
      <td>NaN</td>
      <td>2861.00</td>
      <td>2647.00</td>
      <td>118.00</td>
      <td>2067.00</td>
      <td>31.00</td>
      <td>20119632.00</td>
    </tr>
    <tr>
      <th>counts</th>
      <td>59813</td>
      <td>59813</td>
      <td>59813</td>
      <td>59813</td>
      <td>59813</td>
      <td>59813</td>
      <td>59813</td>
      <td>59813</td>
      <td>59813</td>
      <td>59813</td>
      <td>59813</td>
    </tr>
    <tr>
      <th>uniques</th>
      <td>1989</td>
      <td>968</td>
      <td>2475</td>
      <td>81</td>
      <td>982</td>
      <td>1292</td>
      <td>1051</td>
      <td>88</td>
      <td>109</td>
      <td>31</td>
      <td>373</td>
    </tr>
    <tr>
      <th>missing</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>missing_perc</th>
      <td>0%</td>
      <td>0%</td>
      <td>0%</td>
      <td>0%</td>
      <td>0%</td>
      <td>0%</td>
      <td>0%</td>
      <td>0%</td>
      <td>0%</td>
      <td>0%</td>
      <td>0%</td>
    </tr>
    <tr>
      <th>types</th>
      <td>categorical</td>
      <td>categorical</td>
      <td>numeric</td>
      <td>numeric</td>
      <td>categorical</td>
      <td>numeric</td>
      <td>numeric</td>
      <td>numeric</td>
      <td>numeric</td>
      <td>numeric</td>
      <td>numeric</td>
    </tr>
  </tbody>
</table>
</div>



**Problem definition:**
Predict future behaviour of renters. Will they pay on time or not. If not how late will they be?

# Feature engineering


```python
df.drop(df[df.age < 16].index, inplace=True)

```


```python
df.shape
```




    (58310, 11)




```python
df.paymentDate=pd.to_datetime(df.paymentDate)
```


```python
df['date']=pd.to_datetime(df.paymentDate).dt.to_period('m')
```


```python
df.date.value_counts().min()
```




    41



Not enough data for the last month.


```python
pivoted=df.pivot_table(index='name', columns='date', values='late', aggfunc=np.sum)
```


```python
pivoted.isnull().sum().tail()
```




    date
    2012-06     184
    2012-07     186
    2012-08     204
    2012-09     206
    2012-10    1896
    Freq: M, dtype: int64



Many of the nulls seem to be for the last column so i have decided to drop this column from the remainder of the exercise.


```python
# drop last column
pivoted=pivoted.iloc[:, 0:32]
```


```python
# nan's refer to months where payments not made so setting value to 30
pivoted=pivoted.fillna(30)
```


```python
late=pivoted.mean(axis=1).to_frame()
late.columns=['avg_late']
pd.options.display.float_format = '{:,.0f}'.format
```


```python
data=pd.concat([pivoted,late], axis=1)
```


```python
# create average lateness over the last 6 months

data['6mo']=data.iloc[:,26:32].mean(axis=1)
```


```python
# Create average of last 3 months

data['3mo']=data.iloc[:, 29:32].mean(axis=1)

```


```python
# Create last month column
data['last_mo']=data.iloc[:,31]
```


```python
data=data.reset_index()
```


```python
# create second dataframe with the remaining data

pi2=df.pivot_table(index=['name', 'age', 'rentAmount'], values='late', aggfunc=np.mean).reset_index()
pi2.index.name=pi2.columns.name = None
pi2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>rentAmount</th>
      <th>late</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aaron Huston</td>
      <td>18</td>
      <td>1,720</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Abbey Kluth</td>
      <td>18</td>
      <td>1,619</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Abel Emmett</td>
      <td>44</td>
      <td>1,802</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Abraham Maggi</td>
      <td>18</td>
      <td>1,269</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adah Curnutt</td>
      <td>41</td>
      <td>1,650</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
pi2=pi2.drop_duplicates(['name'], keep='last')
```


```python
pi2.drop('late', axis=1, inplace=True)
```


```python
pi2.shape
```




    (1937, 3)




```python
col_list=['name', 'age', 'rentAmount']
```

486 people have changed apartments!


```python
# Merge the two to create final dataframe and save
data=pd.merge(data, pi2, left_index=True, right_index=True )
data.to_csv('final_dataset.csv')
```


```python
# Start by separating out the regular payers.
```


```python
data['category'] = data['avg_late'].map(lambda x: 0 if x==0 else 1 if x<=8 else 2)
```


```python
fig, axes = plt.subplots(1, 4)

data.hist('last_mo', bins=10, ax=axes[0])
data.hist('3mo', bins=10, ax=axes[1])
data.hist('6mo', bins=10, ax=axes[2])
data.hist('avg_late', bins=10, ax=axes[3])
```




    array([<matplotlib.axes._subplots.AxesSubplot object at 0x1a183ebb38>],
          dtype=object)




![png](output_38_1.png)


I dont have many features so i will skip the feature selection and use the full dataset.


```python
X=data[['avg_late', '3mo','6mo', 'age', 'rentAmount']]
y=data['last_mo']
```


```python
%matplotlib inline
import matplotlib.pyplot as plt
X.hist(bins=15, figsize=(20,15))
plt.savefig("attribute_histogram_plots")
plt.show()
```


![png](output_41_0.png)



```python
sns.set(style="ticks", color_codes=True)
sns.pairplot(X, kind="reg", plot_kws={'line_kws':{'color':'green'}})
```




    <seaborn.axisgrid.PairGrid at 0x1a1c8a7080>




![png](output_42_1.png)



```python
# g = sns.PairGrid(X)
# g.map_diag(sns.kdeplot)
# g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6);
```


```python
corr = X.corr()
corr.style.background_gradient()
```




<style  type="text/css" >
    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row0_col0 {
            background-color:  #023858;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row0_col1 {
            background-color:  #4a98c5;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row0_col2 {
            background-color:  #2484ba;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row0_col3 {
            background-color:  #fcf4fa;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row0_col4 {
            background-color:  #fff7fb;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row1_col0 {
            background-color:  #4897c4;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row1_col1 {
            background-color:  #023858;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row1_col2 {
            background-color:  #034369;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row1_col3 {
            background-color:  #fff7fb;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row1_col4 {
            background-color:  #f9f2f8;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row2_col0 {
            background-color:  #2383ba;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row2_col1 {
            background-color:  #034369;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row2_col2 {
            background-color:  #023858;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row2_col3 {
            background-color:  #fff7fb;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row2_col4 {
            background-color:  #fbf3f9;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row3_col0 {
            background-color:  #fbf3f9;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row3_col1 {
            background-color:  #fff7fb;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row3_col2 {
            background-color:  #fff7fb;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row3_col3 {
            background-color:  #023858;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row3_col4 {
            background-color:  #fef6fb;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row4_col0 {
            background-color:  #fff7fb;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row4_col1 {
            background-color:  #faf3f9;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row4_col2 {
            background-color:  #fdf5fa;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row4_col3 {
            background-color:  #fff7fb;
        }    #T_290f118c_8a6d_11e8_ad2e_de0090e9f601row4_col4 {
            background-color:  #023858;
        }</style>  
<table id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601" > 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >avg_late</th> 
        <th class="col_heading level0 col1" >3mo</th> 
        <th class="col_heading level0 col2" >6mo</th> 
        <th class="col_heading level0 col3" >age</th> 
        <th class="col_heading level0 col4" >rentAmount</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601level0_row0" class="row_heading level0 row0" >avg_late</th> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row0_col0" class="data row0 col0" >1</td> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row0_col1" class="data row0 col1" >0.573477</td> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row0_col2" class="data row0 col2" >0.662202</td> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row0_col3" class="data row0 col3" >-0.00195697</td> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row0_col4" class="data row0 col4" >-0.0301934</td> 
    </tr>    <tr> 
        <th id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601level0_row1" class="row_heading level0 row1" >3mo</th> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row1_col0" class="data row1 col0" >0.573477</td> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row1_col1" class="data row1 col1" >1</td> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row1_col2" class="data row1 col2" >0.959925</td> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row1_col3" class="data row1 col3" >-0.0213324</td> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row1_col4" class="data row1 col4" >0.0135004</td> 
    </tr>    <tr> 
        <th id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601level0_row2" class="row_heading level0 row2" >6mo</th> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row2_col0" class="data row2 col0" >0.662202</td> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row2_col1" class="data row2 col1" >0.959925</td> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row2_col2" class="data row2 col2" >1</td> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row2_col3" class="data row2 col3" >-0.0200836</td> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row2_col4" class="data row2 col4" >-0.00141694</td> 
    </tr>    <tr> 
        <th id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601level0_row3" class="row_heading level0 row3" >age</th> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row3_col0" class="data row3 col0" >-0.00195697</td> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row3_col1" class="data row3 col1" >-0.0213324</td> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row3_col2" class="data row3 col2" >-0.0200836</td> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row3_col3" class="data row3 col3" >1</td> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row3_col4" class="data row3 col4" >-0.0230415</td> 
    </tr>    <tr> 
        <th id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601level0_row4" class="row_heading level0 row4" >rentAmount</th> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row4_col0" class="data row4 col0" >-0.0301934</td> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row4_col1" class="data row4 col1" >0.0135004</td> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row4_col2" class="data row4 col2" >-0.00141694</td> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row4_col3" class="data row4 col3" >-0.0230415</td> 
        <td id="T_290f118c_8a6d_11e8_ad2e_de0090e9f601row4_col4" class="data row4 col4" >1</td> 
    </tr></tbody> 
</table> 




```python
X.plot(kind="scatter", x="avg_late", y="3mo", alpha=0.6, c='rentAmount',cmap=plt.get_cmap("jet"), colorbar=True, sharex=False )
plt.savefig('img.png')
# plt.yscale('log')

plt.savefig('map2.png')
```


![png](output_45_0.png)



```python
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import ensemble
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix
```


```python
# standardize features

ss = StandardScaler()
X = ss.fit_transform(X)
```


```python
# hold out 30% to test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```

Baseline model:


```python
y_baseline = np.mean(y_train)
print('Y baseline:', y_baseline)

data['category'].value_counts().max()/len(data)
```

    Y baseline: 5.21803127874885





    0.7027027027027027




```python
y.median()
```




    0.0




```python
models=[]
models.append(('Linear Regression', LinearRegression()))
models.append(('Random Forrest', RandomForestRegressor()))
models.append(('Gradient Boosting', GradientBoostingRegressor()))
models.append(('Support Vector', SVR()))

results=[]
names=[]

for name, model in models:
    m=model.fit(X_train, y_train)
    score=m.score(X_test, y_test)
    results.append(score)
    names.append(name)
    y_pred = m.predict(X_test)
    model_mse = mean_squared_error(y_pred, y_test)
    model_rmse = np.sqrt(model_mse)
    cvs=cross_val_score(m, X_test, y_test, cv=5)
    predictions=cross_val_predict(m, X_test, y_test)
    
    print(name,'Score:%0.4f'% score)
    print(name,'RMSE: %.4f' % model_rmse)
    print(name, 'CV Accuracy: %0.4f (+/- %0.2f)'% (cvs.mean(), cvs.std()*2))
```

    Linear Regression Score:0.8988
    Linear Regression RMSE: 3.0623
    Linear Regression CV Accuracy: 0.9043 (+/- 0.04)
    Random Forrest Score:0.8943
    Random Forrest RMSE: 3.1297
    Random Forrest CV Accuracy: 0.8878 (+/- 0.09)
    Gradient Boosting Score:0.9013
    Gradient Boosting RMSE: 3.0246
    Gradient Boosting CV Accuracy: 0.8895 (+/- 0.10)
    Support Vector Score:0.8704
    Support Vector RMSE: 3.4661
    Support Vector CV Accuracy: 0.7830 (+/- 0.11)



```python
results
```




    [0.8988475501380275,
     0.8943438987187007,
     0.9013241723545026,
     0.8704125663540474]




```python
GB = GradientBoostingRegressor()
GB.fit(X_train, y_train)
```




    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, min_samples_leaf=1,
                 min_samples_split=2, min_weight_fraction_leaf=0.0,
                 n_estimators=100, presort='auto', random_state=None,
                 subsample=1.0, verbose=0, warm_start=False)




```python
print('Gradient Boosting R squared": %.4f' % GB.score(X_test, y_test))

```

    Gradient Boosting R squared": 0.9033



```python
y_pred = model.predict(X_test)
model_mse = mean_squared_error(y_pred, y_test)
model_rmse = np.sqrt(model_mse)
print('Gradient Boosting RMSE: %.4f' % model_rmse)
```

    Gradient Boosting RMSE: 2.6107



```python
feature_labels = np.array(['avg_late', '3mo','last_2mo', 'age', 'rentAmount' ])

importance = GB.feature_importances_
feature_indexes_by_importance = importance.argsort()
for index in feature_indexes_by_importance:
    print('{}-{:.2f}%'.format(feature_labels[index], (importance[index] *100.0)))
```

    age-4.14%
    last_2mo-16.76%
    rentAmount-19.07%
    avg_late-22.80%
    3mo-37.23%



```python
plt.style.use('ggplot')
fig, ax=plt.subplots(figsize=(10,5))
ax = sns.barplot(x=feature_indexes_by_importance, y=importance)
plt.show()
```


![png](output_58_0.png)

