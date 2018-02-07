
# Enron Fraud Exploration with Machine Learning 

Jake Neyer, Feb 2018

## Abstract

The goal of this project is to use machine learning tactics to indentify persons-of-interest in a currated dataset of emails which were released after the Enron Scandal. Developing patterns of fraud and malicious intent in a dataset this large is nearly impossible by simple obervation and human intuition which is why machine learning tactics are essential.


## Background

The Enron scandal that was publicized in October 2001 was perhaps one of the greatest examples of corporate fraud in American history. Through several means of hiding billions of dollars in debt, Enron executives were able to artificially increase and maintain stock value of the compaby. Disgruntled shareholders filed a lawsuit against the company and in December 2, 2001 which led to Enron filing for bankruptcy. Several Enron executives were indicted and sentenced. Addionally, Authur Andersen(a large audit and accounting partnership) was found guilty of illegally destroying documents related to the investigation and ultimately closed its doors because of it.

Read more here: https://en.wikipedia.org/wiki/Enron_scandal

## Loading Dataset


```python
#!/usr/bin/python
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import tree
import pandas
import matplotlib.pyplot as plt
```


```python
### Loading the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
```

Translating the dictionary into a pandas dataframe will make exploratory data analysis easier.


```python
#Translating Data in pandas data frame
df = pandas.DataFrame.from_records(list(data_dict.values()))
employees = pandas.Series(list(data_dict.keys()))
# set the index of df to be the employees series:
df.set_index(employees, inplace=True)
```

## Exploring Dataset

Taking a look at our POIs attributes


```python
#Dataframe of POIs
poi = df[df['poi'] == 1]
poi

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
      <th>bonus</th>
      <th>deferral_payments</th>
      <th>deferred_income</th>
      <th>director_fees</th>
      <th>email_address</th>
      <th>exercised_stock_options</th>
      <th>expenses</th>
      <th>from_messages</th>
      <th>from_poi_to_this_person</th>
      <th>from_this_person_to_poi</th>
      <th>...</th>
      <th>long_term_incentive</th>
      <th>other</th>
      <th>poi</th>
      <th>restricted_stock</th>
      <th>restricted_stock_deferred</th>
      <th>salary</th>
      <th>shared_receipt_with_poi</th>
      <th>to_messages</th>
      <th>total_payments</th>
      <th>total_stock_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>HANNON KEVIN P</th>
      <td>1500000</td>
      <td>NaN</td>
      <td>-3117011</td>
      <td>NaN</td>
      <td>kevin.hannon@enron.com</td>
      <td>5538001</td>
      <td>34039</td>
      <td>32</td>
      <td>32</td>
      <td>21</td>
      <td>...</td>
      <td>1617011</td>
      <td>11350</td>
      <td>True</td>
      <td>853064</td>
      <td>NaN</td>
      <td>243293</td>
      <td>1035</td>
      <td>1045</td>
      <td>288682</td>
      <td>6391065</td>
    </tr>
    <tr>
      <th>COLWELL WESLEY</th>
      <td>1200000</td>
      <td>27610</td>
      <td>-144062</td>
      <td>NaN</td>
      <td>wes.colwell@enron.com</td>
      <td>NaN</td>
      <td>16514</td>
      <td>40</td>
      <td>240</td>
      <td>11</td>
      <td>...</td>
      <td>NaN</td>
      <td>101740</td>
      <td>True</td>
      <td>698242</td>
      <td>NaN</td>
      <td>288542</td>
      <td>1132</td>
      <td>1758</td>
      <td>1490344</td>
      <td>698242</td>
    </tr>
    <tr>
      <th>RIEKER PAULA H</th>
      <td>700000</td>
      <td>214678</td>
      <td>-100000</td>
      <td>NaN</td>
      <td>paula.rieker@enron.com</td>
      <td>1635238</td>
      <td>33271</td>
      <td>82</td>
      <td>35</td>
      <td>48</td>
      <td>...</td>
      <td>NaN</td>
      <td>1950</td>
      <td>True</td>
      <td>283649</td>
      <td>NaN</td>
      <td>249201</td>
      <td>1258</td>
      <td>1328</td>
      <td>1099100</td>
      <td>1918887</td>
    </tr>
    <tr>
      <th>KOPPER MICHAEL J</th>
      <td>800000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>michael.kopper@enron.com</td>
      <td>NaN</td>
      <td>118134</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>602671</td>
      <td>907502</td>
      <td>True</td>
      <td>985032</td>
      <td>NaN</td>
      <td>224305</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2652612</td>
      <td>985032</td>
    </tr>
    <tr>
      <th>SHELBY REX</th>
      <td>200000</td>
      <td>NaN</td>
      <td>-4167</td>
      <td>NaN</td>
      <td>rex.shelby@enron.com</td>
      <td>1624396</td>
      <td>22884</td>
      <td>39</td>
      <td>13</td>
      <td>14</td>
      <td>...</td>
      <td>NaN</td>
      <td>1573324</td>
      <td>True</td>
      <td>869220</td>
      <td>NaN</td>
      <td>211844</td>
      <td>91</td>
      <td>225</td>
      <td>2003885</td>
      <td>2493616</td>
    </tr>
    <tr>
      <th>DELAINEY DAVID W</th>
      <td>3000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>david.delainey@enron.com</td>
      <td>2291113</td>
      <td>86174</td>
      <td>3069</td>
      <td>66</td>
      <td>609</td>
      <td>...</td>
      <td>1294981</td>
      <td>1661</td>
      <td>True</td>
      <td>1323148</td>
      <td>NaN</td>
      <td>365163</td>
      <td>2097</td>
      <td>3093</td>
      <td>4747979</td>
      <td>3614261</td>
    </tr>
    <tr>
      <th>LAY KENNETH L</th>
      <td>7000000</td>
      <td>202911</td>
      <td>-300000</td>
      <td>NaN</td>
      <td>kenneth.lay@enron.com</td>
      <td>34348384</td>
      <td>99832</td>
      <td>36</td>
      <td>123</td>
      <td>16</td>
      <td>...</td>
      <td>3600000</td>
      <td>10359729</td>
      <td>True</td>
      <td>14761694</td>
      <td>NaN</td>
      <td>1072321</td>
      <td>2411</td>
      <td>4273</td>
      <td>103559793</td>
      <td>49110078</td>
    </tr>
    <tr>
      <th>BOWEN JR RAYMOND M</th>
      <td>1350000</td>
      <td>NaN</td>
      <td>-833</td>
      <td>NaN</td>
      <td>raymond.bowen@enron.com</td>
      <td>NaN</td>
      <td>65907</td>
      <td>27</td>
      <td>140</td>
      <td>15</td>
      <td>...</td>
      <td>974293</td>
      <td>1621</td>
      <td>True</td>
      <td>252055</td>
      <td>NaN</td>
      <td>278601</td>
      <td>1593</td>
      <td>1858</td>
      <td>2669589</td>
      <td>252055</td>
    </tr>
    <tr>
      <th>BELDEN TIMOTHY N</th>
      <td>5249999</td>
      <td>2144013</td>
      <td>-2334434</td>
      <td>NaN</td>
      <td>tim.belden@enron.com</td>
      <td>953136</td>
      <td>17355</td>
      <td>484</td>
      <td>228</td>
      <td>108</td>
      <td>...</td>
      <td>NaN</td>
      <td>210698</td>
      <td>True</td>
      <td>157569</td>
      <td>NaN</td>
      <td>213999</td>
      <td>5521</td>
      <td>7991</td>
      <td>5501630</td>
      <td>1110705</td>
    </tr>
    <tr>
      <th>FASTOW ANDREW S</th>
      <td>1300000</td>
      <td>NaN</td>
      <td>-1386055</td>
      <td>NaN</td>
      <td>andrew.fastow@enron.com</td>
      <td>NaN</td>
      <td>55921</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1736055</td>
      <td>277464</td>
      <td>True</td>
      <td>1794412</td>
      <td>NaN</td>
      <td>440698</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2424083</td>
      <td>1794412</td>
    </tr>
    <tr>
      <th>CALGER CHRISTOPHER F</th>
      <td>1250000</td>
      <td>NaN</td>
      <td>-262500</td>
      <td>NaN</td>
      <td>christopher.calger@enron.com</td>
      <td>NaN</td>
      <td>35818</td>
      <td>144</td>
      <td>199</td>
      <td>25</td>
      <td>...</td>
      <td>375304</td>
      <td>486</td>
      <td>True</td>
      <td>126027</td>
      <td>NaN</td>
      <td>240189</td>
      <td>2188</td>
      <td>2598</td>
      <td>1639297</td>
      <td>126027</td>
    </tr>
    <tr>
      <th>RICE KENNETH D</th>
      <td>1750000</td>
      <td>NaN</td>
      <td>-3504386</td>
      <td>NaN</td>
      <td>ken.rice@enron.com</td>
      <td>19794175</td>
      <td>46950</td>
      <td>18</td>
      <td>42</td>
      <td>4</td>
      <td>...</td>
      <td>1617011</td>
      <td>174839</td>
      <td>True</td>
      <td>2748364</td>
      <td>NaN</td>
      <td>420636</td>
      <td>864</td>
      <td>905</td>
      <td>505050</td>
      <td>22542539</td>
    </tr>
    <tr>
      <th>SKILLING JEFFREY K</th>
      <td>5600000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>jeff.skilling@enron.com</td>
      <td>19250000</td>
      <td>29336</td>
      <td>108</td>
      <td>88</td>
      <td>30</td>
      <td>...</td>
      <td>1920000</td>
      <td>22122</td>
      <td>True</td>
      <td>6843672</td>
      <td>NaN</td>
      <td>1111258</td>
      <td>2042</td>
      <td>3627</td>
      <td>8682716</td>
      <td>26093672</td>
    </tr>
    <tr>
      <th>YEAGER F SCOTT</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>scott.yeager@enron.com</td>
      <td>8308552</td>
      <td>53947</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>147950</td>
      <td>True</td>
      <td>3576206</td>
      <td>NaN</td>
      <td>158403</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>360300</td>
      <td>11884758</td>
    </tr>
    <tr>
      <th>HIRKO JOSEPH</th>
      <td>NaN</td>
      <td>10259</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>joe.hirko@enron.com</td>
      <td>30766064</td>
      <td>77978</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>2856</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91093</td>
      <td>30766064</td>
    </tr>
    <tr>
      <th>KOENIG MARK E</th>
      <td>700000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>mark.koenig@enron.com</td>
      <td>671737</td>
      <td>127017</td>
      <td>61</td>
      <td>53</td>
      <td>15</td>
      <td>...</td>
      <td>300000</td>
      <td>150458</td>
      <td>True</td>
      <td>1248318</td>
      <td>NaN</td>
      <td>309946</td>
      <td>2271</td>
      <td>2374</td>
      <td>1587421</td>
      <td>1920055</td>
    </tr>
    <tr>
      <th>CAUSEY RICHARD A</th>
      <td>1000000</td>
      <td>NaN</td>
      <td>-235000</td>
      <td>NaN</td>
      <td>richard.causey@enron.com</td>
      <td>NaN</td>
      <td>30674</td>
      <td>49</td>
      <td>58</td>
      <td>12</td>
      <td>...</td>
      <td>350000</td>
      <td>307895</td>
      <td>True</td>
      <td>2502063</td>
      <td>NaN</td>
      <td>415189</td>
      <td>1585</td>
      <td>1892</td>
      <td>1868758</td>
      <td>2502063</td>
    </tr>
    <tr>
      <th>GLISAN JR BEN F</th>
      <td>600000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ben.glisan@enron.com</td>
      <td>384728</td>
      <td>125978</td>
      <td>16</td>
      <td>52</td>
      <td>6</td>
      <td>...</td>
      <td>71023</td>
      <td>200308</td>
      <td>True</td>
      <td>393818</td>
      <td>NaN</td>
      <td>274975</td>
      <td>874</td>
      <td>873</td>
      <td>1272284</td>
      <td>778546</td>
    </tr>
  </tbody>
</table>
<p>18 rows × 21 columns</p>
</div>



And now taking a look at the Non-POI attributes.


```python
#Dataframe of Non-POIs
not_poi = df[df['poi'] == 0]
not_poi
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
      <th>bonus</th>
      <th>deferral_payments</th>
      <th>deferred_income</th>
      <th>director_fees</th>
      <th>email_address</th>
      <th>exercised_stock_options</th>
      <th>expenses</th>
      <th>from_messages</th>
      <th>from_poi_to_this_person</th>
      <th>from_this_person_to_poi</th>
      <th>...</th>
      <th>long_term_incentive</th>
      <th>other</th>
      <th>poi</th>
      <th>restricted_stock</th>
      <th>restricted_stock_deferred</th>
      <th>salary</th>
      <th>shared_receipt_with_poi</th>
      <th>to_messages</th>
      <th>total_payments</th>
      <th>total_stock_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>METTS MARK</th>
      <td>600000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>mark.metts@enron.com</td>
      <td>NaN</td>
      <td>94299</td>
      <td>29</td>
      <td>38</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>1740</td>
      <td>False</td>
      <td>585062</td>
      <td>NaN</td>
      <td>365788</td>
      <td>702</td>
      <td>807</td>
      <td>1061827</td>
      <td>585062</td>
    </tr>
    <tr>
      <th>BAXTER JOHN C</th>
      <td>1200000</td>
      <td>1295738</td>
      <td>-1386055</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6680544</td>
      <td>11200</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1586055</td>
      <td>2660303</td>
      <td>False</td>
      <td>3942714</td>
      <td>NaN</td>
      <td>267102</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5634343</td>
      <td>10623258</td>
    </tr>
    <tr>
      <th>ELLIOTT STEVEN</th>
      <td>350000</td>
      <td>NaN</td>
      <td>-400729</td>
      <td>NaN</td>
      <td>steven.elliott@enron.com</td>
      <td>4890344</td>
      <td>78552</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>12961</td>
      <td>False</td>
      <td>1788391</td>
      <td>NaN</td>
      <td>170941</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>211725</td>
      <td>6678735</td>
    </tr>
    <tr>
      <th>CORDES WILLIAM R</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>bill.cordes@enron.com</td>
      <td>651850</td>
      <td>NaN</td>
      <td>12</td>
      <td>10</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>386335</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>58</td>
      <td>764</td>
      <td>NaN</td>
      <td>1038185</td>
    </tr>
    <tr>
      <th>MORDAUNT KRISTINA M</th>
      <td>325000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>kristina.mordaunt@enron.com</td>
      <td>NaN</td>
      <td>35018</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>1411</td>
      <td>False</td>
      <td>208510</td>
      <td>NaN</td>
      <td>267093</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>628522</td>
      <td>208510</td>
    </tr>
    <tr>
      <th>MEYER ROCKFORD G</th>
      <td>NaN</td>
      <td>1848227</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>rockford.meyer@enron.com</td>
      <td>493489</td>
      <td>NaN</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>462384</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22</td>
      <td>232</td>
      <td>1848227</td>
      <td>955873</td>
    </tr>
    <tr>
      <th>MCMAHON JEFFREY</th>
      <td>2600000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>jeffrey.mcmahon@enron.com</td>
      <td>1104054</td>
      <td>137108</td>
      <td>48</td>
      <td>58</td>
      <td>26</td>
      <td>...</td>
      <td>694862</td>
      <td>297353</td>
      <td>False</td>
      <td>558801</td>
      <td>NaN</td>
      <td>370448</td>
      <td>2228</td>
      <td>2355</td>
      <td>4099771</td>
      <td>1662855</td>
    </tr>
    <tr>
      <th>HORTON STANLEY C</th>
      <td>NaN</td>
      <td>3131860</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>stanley.horton@enron.com</td>
      <td>5210569</td>
      <td>NaN</td>
      <td>1073</td>
      <td>44</td>
      <td>15</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>2046079</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1074</td>
      <td>2350</td>
      <td>3131860</td>
      <td>7256648</td>
    </tr>
    <tr>
      <th>PIPER GREGORY F</th>
      <td>400000</td>
      <td>1130036</td>
      <td>-33333</td>
      <td>NaN</td>
      <td>greg.piper@enron.com</td>
      <td>880290</td>
      <td>43057</td>
      <td>222</td>
      <td>61</td>
      <td>48</td>
      <td>...</td>
      <td>NaN</td>
      <td>778</td>
      <td>False</td>
      <td>409554</td>
      <td>-409554</td>
      <td>197091</td>
      <td>742</td>
      <td>1238</td>
      <td>1737629</td>
      <td>880290</td>
    </tr>
    <tr>
      <th>HUMPHREY GENE E</th>
      <td>NaN</td>
      <td>2964506</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>gene.humphrey@enron.com</td>
      <td>2282768</td>
      <td>4994</td>
      <td>17</td>
      <td>10</td>
      <td>17</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>130724</td>
      <td>119</td>
      <td>128</td>
      <td>3100224</td>
      <td>2282768</td>
    </tr>
    <tr>
      <th>UMANOFF ADAM S</th>
      <td>788750</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>adam.umanoff@enron.com</td>
      <td>NaN</td>
      <td>53122</td>
      <td>18</td>
      <td>12</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>288589</td>
      <td>41</td>
      <td>111</td>
      <td>1130461</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BLACHMAN JEREMY M</th>
      <td>850000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>jeremy.blachman@enron.com</td>
      <td>765313</td>
      <td>84208</td>
      <td>14</td>
      <td>25</td>
      <td>2</td>
      <td>...</td>
      <td>831809</td>
      <td>272</td>
      <td>False</td>
      <td>189041</td>
      <td>NaN</td>
      <td>248546</td>
      <td>2326</td>
      <td>2475</td>
      <td>2014835</td>
      <td>954354</td>
    </tr>
    <tr>
      <th>SUNDE MARTIN</th>
      <td>700000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>marty.sunde@enron.com</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>38</td>
      <td>37</td>
      <td>13</td>
      <td>...</td>
      <td>476451</td>
      <td>111122</td>
      <td>False</td>
      <td>698920</td>
      <td>NaN</td>
      <td>257486</td>
      <td>2565</td>
      <td>2647</td>
      <td>1545059</td>
      <td>698920</td>
    </tr>
    <tr>
      <th>GIBBS DANA R</th>
      <td>NaN</td>
      <td>504610</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>dana.gibbs@enron.com</td>
      <td>2218275</td>
      <td>NaN</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>461912</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23</td>
      <td>169</td>
      <td>966522</td>
      <td>2218275</td>
    </tr>
    <tr>
      <th>LOWRY CHARLES P</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>372205</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>153686</td>
      <td>-153686</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>372205</td>
    </tr>
    <tr>
      <th>MULLER MARK S</th>
      <td>1100000</td>
      <td>842924</td>
      <td>-719000</td>
      <td>NaN</td>
      <td>s..muller@enron.com</td>
      <td>1056320</td>
      <td>NaN</td>
      <td>16</td>
      <td>12</td>
      <td>0</td>
      <td>...</td>
      <td>1725545</td>
      <td>947</td>
      <td>False</td>
      <td>360528</td>
      <td>NaN</td>
      <td>251654</td>
      <td>114</td>
      <td>136</td>
      <td>3202070</td>
      <td>1416848</td>
    </tr>
    <tr>
      <th>JACKSON CHARLENE R</th>
      <td>250000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>charlene.jackson@enron.com</td>
      <td>185063</td>
      <td>10181</td>
      <td>56</td>
      <td>25</td>
      <td>19</td>
      <td>...</td>
      <td>NaN</td>
      <td>2435</td>
      <td>False</td>
      <td>540672</td>
      <td>NaN</td>
      <td>288558</td>
      <td>117</td>
      <td>258</td>
      <td>551174</td>
      <td>725735</td>
    </tr>
    <tr>
      <th>WESTFAHL RICHARD K</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>-10800</td>
      <td>NaN</td>
      <td>dick.westfahl@enron.com</td>
      <td>NaN</td>
      <td>51870</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>256191</td>
      <td>401130</td>
      <td>False</td>
      <td>384930</td>
      <td>NaN</td>
      <td>63744</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>762135</td>
      <td>384930</td>
    </tr>
    <tr>
      <th>WALTERS GARETH W</th>
      <td>NaN</td>
      <td>53625</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1030329</td>
      <td>33785</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>87410</td>
      <td>1030329</td>
    </tr>
    <tr>
      <th>WALLS JR ROBERT H</th>
      <td>850000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>rob.walls@enron.com</td>
      <td>4346544</td>
      <td>50936</td>
      <td>146</td>
      <td>17</td>
      <td>0</td>
      <td>...</td>
      <td>540751</td>
      <td>2</td>
      <td>False</td>
      <td>1552453</td>
      <td>NaN</td>
      <td>357091</td>
      <td>215</td>
      <td>671</td>
      <td>1798780</td>
      <td>5898997</td>
    </tr>
    <tr>
      <th>KITCHEN LOUISE</th>
      <td>3100000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>louise.kitchen@enron.com</td>
      <td>81042</td>
      <td>5774</td>
      <td>1728</td>
      <td>251</td>
      <td>194</td>
      <td>...</td>
      <td>NaN</td>
      <td>93925</td>
      <td>False</td>
      <td>466101</td>
      <td>NaN</td>
      <td>271442</td>
      <td>3669</td>
      <td>8305</td>
      <td>3471141</td>
      <td>547143</td>
    </tr>
    <tr>
      <th>CHAN RONNIE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>-98784</td>
      <td>98784</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>32460</td>
      <td>-32460</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BELFER ROBERT</th>
      <td>NaN</td>
      <td>-102500</td>
      <td>NaN</td>
      <td>3285</td>
      <td>NaN</td>
      <td>3285</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>44093</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>102500</td>
      <td>-44093</td>
    </tr>
    <tr>
      <th>SHANKMAN JEFFREY A</th>
      <td>2000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>jeffrey.shankman@enron.com</td>
      <td>1441898</td>
      <td>178979</td>
      <td>2681</td>
      <td>94</td>
      <td>83</td>
      <td>...</td>
      <td>554422</td>
      <td>1191</td>
      <td>False</td>
      <td>630137</td>
      <td>NaN</td>
      <td>304110</td>
      <td>1730</td>
      <td>3221</td>
      <td>3038702</td>
      <td>2072035</td>
    </tr>
    <tr>
      <th>WODRASKA JOHN</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>john.wodraska@enron.com</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>189583</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>189583</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BERGSIEKER RICHARD P</th>
      <td>250000</td>
      <td>NaN</td>
      <td>-485813</td>
      <td>NaN</td>
      <td>rick.bergsieker@enron.com</td>
      <td>NaN</td>
      <td>59175</td>
      <td>59</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>180250</td>
      <td>427316</td>
      <td>False</td>
      <td>659249</td>
      <td>NaN</td>
      <td>187922</td>
      <td>233</td>
      <td>383</td>
      <td>618850</td>
      <td>659249</td>
    </tr>
    <tr>
      <th>URQUHART JOHN A</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>-36666</td>
      <td>36666</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>228656</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>228656</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BIBI PHILIPPE A</th>
      <td>1000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>philippe.bibi@enron.com</td>
      <td>1465734</td>
      <td>38559</td>
      <td>40</td>
      <td>23</td>
      <td>8</td>
      <td>...</td>
      <td>369721</td>
      <td>425688</td>
      <td>False</td>
      <td>378082</td>
      <td>NaN</td>
      <td>213625</td>
      <td>1336</td>
      <td>1607</td>
      <td>2047593</td>
      <td>1843816</td>
    </tr>
    <tr>
      <th>WHALEY DAVID A</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>98718</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>98718</td>
    </tr>
    <tr>
      <th>BECK SALLY W</th>
      <td>700000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sally.beck@enron.com</td>
      <td>NaN</td>
      <td>37172</td>
      <td>4343</td>
      <td>144</td>
      <td>386</td>
      <td>...</td>
      <td>NaN</td>
      <td>566</td>
      <td>False</td>
      <td>126027</td>
      <td>NaN</td>
      <td>231330</td>
      <td>2639</td>
      <td>7315</td>
      <td>969068</td>
      <td>126027</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>JAEDICKE ROBERT</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>-25000</td>
      <td>108750</td>
      <td>NaN</td>
      <td>431750</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>44093</td>
      <td>-44093</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>83750</td>
      <td>431750</td>
    </tr>
    <tr>
      <th>WINOKUR JR. HERBERT S</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>-25000</td>
      <td>108579</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1413</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>84992</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BROWN MICHAEL</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>michael.brown@enron.com</td>
      <td>NaN</td>
      <td>49288</td>
      <td>41</td>
      <td>13</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>761</td>
      <td>1486</td>
      <td>49288</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BADUM JAMES P</th>
      <td>NaN</td>
      <td>178980</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>257817</td>
      <td>3486</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>182466</td>
      <td>257817</td>
    </tr>
    <tr>
      <th>HUGHES JAMES A</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>james.hughes@enron.com</td>
      <td>754966</td>
      <td>NaN</td>
      <td>34</td>
      <td>35</td>
      <td>5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>363428</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>589</td>
      <td>719</td>
      <td>NaN</td>
      <td>1118394</td>
    </tr>
    <tr>
      <th>REYNOLDS LAWRENCE</th>
      <td>100000</td>
      <td>51365</td>
      <td>-200000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4160672</td>
      <td>8409</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>156250</td>
      <td>202052</td>
      <td>False</td>
      <td>201483</td>
      <td>-140264</td>
      <td>76399</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>394475</td>
      <td>4221891</td>
    </tr>
    <tr>
      <th>DIMICHELE RICHARD G</th>
      <td>1000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>richard.dimichele@enron.com</td>
      <td>8191755</td>
      <td>35812</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>694862</td>
      <td>374689</td>
      <td>False</td>
      <td>126027</td>
      <td>NaN</td>
      <td>262788</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2368151</td>
      <td>8317782</td>
    </tr>
    <tr>
      <th>BHATNAGAR SANJAY</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>137864</td>
      <td>sanjay.bhatnagar@enron.com</td>
      <td>2604490</td>
      <td>NaN</td>
      <td>29</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>137864</td>
      <td>False</td>
      <td>-2604490</td>
      <td>15456290</td>
      <td>NaN</td>
      <td>463</td>
      <td>523</td>
      <td>15456290</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CARTER REBECCA C</th>
      <td>300000</td>
      <td>NaN</td>
      <td>-159792</td>
      <td>NaN</td>
      <td>rebecca.carter@enron.com</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>29</td>
      <td>7</td>
      <td>...</td>
      <td>75000</td>
      <td>540</td>
      <td>False</td>
      <td>307301</td>
      <td>-307301</td>
      <td>261809</td>
      <td>196</td>
      <td>312</td>
      <td>477557</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BUCHANAN HAROLD G</th>
      <td>500000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>john.buchanan@enron.com</td>
      <td>825464</td>
      <td>600</td>
      <td>125</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>304805</td>
      <td>1215</td>
      <td>False</td>
      <td>189041</td>
      <td>NaN</td>
      <td>248017</td>
      <td>23</td>
      <td>1088</td>
      <td>1054637</td>
      <td>1014505</td>
    </tr>
    <tr>
      <th>YEAP SOON</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>192758</td>
      <td>55097</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>55097</td>
      <td>192758</td>
    </tr>
    <tr>
      <th>MURRAY JULIA H</th>
      <td>400000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>julia.murray@enron.com</td>
      <td>400478</td>
      <td>57580</td>
      <td>45</td>
      <td>11</td>
      <td>2</td>
      <td>...</td>
      <td>125000</td>
      <td>330</td>
      <td>False</td>
      <td>196983</td>
      <td>NaN</td>
      <td>229284</td>
      <td>395</td>
      <td>2192</td>
      <td>812194</td>
      <td>597461</td>
    </tr>
    <tr>
      <th>GARLAND C KEVIN</th>
      <td>850000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>kevin.garland@enron.com</td>
      <td>636246</td>
      <td>48405</td>
      <td>44</td>
      <td>10</td>
      <td>27</td>
      <td>...</td>
      <td>375304</td>
      <td>60814</td>
      <td>False</td>
      <td>259907</td>
      <td>NaN</td>
      <td>231946</td>
      <td>178</td>
      <td>209</td>
      <td>1566469</td>
      <td>896153</td>
    </tr>
    <tr>
      <th>DODSON KEITH</th>
      <td>70000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>keith.dodson@enron.com</td>
      <td>NaN</td>
      <td>28164</td>
      <td>14</td>
      <td>10</td>
      <td>3</td>
      <td>...</td>
      <td>NaN</td>
      <td>774</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>221003</td>
      <td>114</td>
      <td>176</td>
      <td>319941</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>DIETRICH JANET R</th>
      <td>600000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>janet.dietrich@enron.com</td>
      <td>1550019</td>
      <td>3475</td>
      <td>63</td>
      <td>305</td>
      <td>14</td>
      <td>...</td>
      <td>556416</td>
      <td>473</td>
      <td>False</td>
      <td>315068</td>
      <td>NaN</td>
      <td>250100</td>
      <td>1902</td>
      <td>2572</td>
      <td>1410464</td>
      <td>1865087</td>
    </tr>
    <tr>
      <th>DERRICK JR. JAMES V</th>
      <td>800000</td>
      <td>NaN</td>
      <td>-1284000</td>
      <td>NaN</td>
      <td>james.derrick@enron.com</td>
      <td>8831913</td>
      <td>51124</td>
      <td>909</td>
      <td>64</td>
      <td>20</td>
      <td>...</td>
      <td>484000</td>
      <td>7482</td>
      <td>False</td>
      <td>1787380</td>
      <td>-1787380</td>
      <td>492375</td>
      <td>1401</td>
      <td>2181</td>
      <td>550981</td>
      <td>8831913</td>
    </tr>
    <tr>
      <th>FREVERT MARK A</th>
      <td>2000000</td>
      <td>6426990</td>
      <td>-3367011</td>
      <td>NaN</td>
      <td>mark.frevert@enron.com</td>
      <td>10433518</td>
      <td>86987</td>
      <td>21</td>
      <td>242</td>
      <td>6</td>
      <td>...</td>
      <td>1617011</td>
      <td>7427621</td>
      <td>False</td>
      <td>4188667</td>
      <td>NaN</td>
      <td>1060932</td>
      <td>2979</td>
      <td>3275</td>
      <td>17252530</td>
      <td>14622185</td>
    </tr>
    <tr>
      <th>PAI LOU L</th>
      <td>1000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>lou.pai@enron.com</td>
      <td>15364167</td>
      <td>32047</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>1829457</td>
      <td>False</td>
      <td>8453763</td>
      <td>NaN</td>
      <td>261879</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3123383</td>
      <td>23817930</td>
    </tr>
    <tr>
      <th>BAY FRANKLIN R</th>
      <td>400000</td>
      <td>260455</td>
      <td>-201641</td>
      <td>NaN</td>
      <td>frank.bay@enron.com</td>
      <td>NaN</td>
      <td>129142</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>69</td>
      <td>False</td>
      <td>145796</td>
      <td>-82782</td>
      <td>239671</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>827696</td>
      <td>63014</td>
    </tr>
    <tr>
      <th>HAYSLETT RODERICK J</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>rod.hayslett@enron.com</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1061</td>
      <td>35</td>
      <td>38</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>346663</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>571</td>
      <td>2649</td>
      <td>NaN</td>
      <td>346663</td>
    </tr>
    <tr>
      <th>FUGH JOHN L</th>
      <td>NaN</td>
      <td>50591</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>176378</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>50591</td>
      <td>176378</td>
    </tr>
    <tr>
      <th>FALLON JAMES B</th>
      <td>2500000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>jim.fallon@enron.com</td>
      <td>940257</td>
      <td>95924</td>
      <td>75</td>
      <td>42</td>
      <td>37</td>
      <td>...</td>
      <td>374347</td>
      <td>401481</td>
      <td>False</td>
      <td>1392142</td>
      <td>NaN</td>
      <td>304588</td>
      <td>1604</td>
      <td>1755</td>
      <td>3676340</td>
      <td>2332399</td>
    </tr>
    <tr>
      <th>SAVAGE FRANK</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>-121284</td>
      <td>125034</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3750</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>IZZO LAWRENCE L</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>larry.izzo@enron.com</td>
      <td>2165172</td>
      <td>28093</td>
      <td>19</td>
      <td>28</td>
      <td>5</td>
      <td>...</td>
      <td>312500</td>
      <td>1553729</td>
      <td>False</td>
      <td>3654808</td>
      <td>NaN</td>
      <td>85274</td>
      <td>437</td>
      <td>496</td>
      <td>1979596</td>
      <td>5819980</td>
    </tr>
    <tr>
      <th>TILNEY ELIZABETH A</th>
      <td>300000</td>
      <td>NaN</td>
      <td>-575000</td>
      <td>NaN</td>
      <td>elizabeth.tilney@enron.com</td>
      <td>591250</td>
      <td>NaN</td>
      <td>19</td>
      <td>10</td>
      <td>11</td>
      <td>...</td>
      <td>275000</td>
      <td>152055</td>
      <td>False</td>
      <td>576792</td>
      <td>NaN</td>
      <td>247338</td>
      <td>379</td>
      <td>460</td>
      <td>399393</td>
      <td>1168042</td>
    </tr>
    <tr>
      <th>MARTIN AMANDA K</th>
      <td>NaN</td>
      <td>85430</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>a..martin@enron.com</td>
      <td>2070306</td>
      <td>8211</td>
      <td>230</td>
      <td>8</td>
      <td>0</td>
      <td>...</td>
      <td>5145434</td>
      <td>2818454</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>349487</td>
      <td>477</td>
      <td>1522</td>
      <td>8407016</td>
      <td>2070306</td>
    </tr>
    <tr>
      <th>BUY RICHARD B</th>
      <td>900000</td>
      <td>649584</td>
      <td>-694862</td>
      <td>NaN</td>
      <td>rick.buy@enron.com</td>
      <td>2542813</td>
      <td>NaN</td>
      <td>1053</td>
      <td>156</td>
      <td>71</td>
      <td>...</td>
      <td>769862</td>
      <td>400572</td>
      <td>False</td>
      <td>901657</td>
      <td>NaN</td>
      <td>330546</td>
      <td>2333</td>
      <td>3523</td>
      <td>2355702</td>
      <td>3444470</td>
    </tr>
    <tr>
      <th>GRAMM WENDY L</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>119292</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>119292</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TAYLOR MITCHELL S</th>
      <td>600000</td>
      <td>227449</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>mitchell.taylor@enron.com</td>
      <td>3181250</td>
      <td>NaN</td>
      <td>29</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>563798</td>
      <td>NaN</td>
      <td>265214</td>
      <td>300</td>
      <td>533</td>
      <td>1092663</td>
      <td>3745048</td>
    </tr>
    <tr>
      <th>DONAHUE JR JEFFREY M</th>
      <td>800000</td>
      <td>NaN</td>
      <td>-300000</td>
      <td>NaN</td>
      <td>jeff.donahue@enron.com</td>
      <td>765920</td>
      <td>96268</td>
      <td>22</td>
      <td>188</td>
      <td>11</td>
      <td>...</td>
      <td>NaN</td>
      <td>891</td>
      <td>False</td>
      <td>315068</td>
      <td>NaN</td>
      <td>278601</td>
      <td>772</td>
      <td>865</td>
      <td>875760</td>
      <td>1080988</td>
    </tr>
  </tbody>
</table>
<p>128 rows × 21 columns</p>
</div>




```python
#Financial Averages
#
#Average POI Salary
print "Average POI Salary: ", poi["salary"].astype(float).mean()
#Average Non-POI Salary
print "Average Non-POI Salary: ", not_poi["salary"].astype(float).mean(), "\n"

#Average POI Bonus
print "Average POI Bonus: ", poi["bonus"].astype(float).mean()
#Average Non-POI Bonus
print "Average Non-POI Bonus: ", not_poi["bonus"].astype(float).mean(), "\n"

#Average POI Bonus
print "Average POI total payments: ", poi["total_payments"].astype(float).mean()
#Average Non-POI Bonus
print "Average Non-POI total payments: ", not_poi["total_payments"].astype(float).mean(), "\n"

#Average POI Total Stock Value
print "Average POI Total Stock Value: ", poi["total_stock_value"].astype(float).mean()
#Average Non-POI Total Stock Value
print "Average Non-POI Total Stock Value: ", not_poi["total_stock_value"].astype(float).mean(), "\n"

#Email Averages
#
#Average Emails From POI
print "Average Emails From POI to POI: ", poi["from_poi_to_this_person"].astype(float).mean()
#Average Emails From POI (Non-POIs)
print "Average Emails From POI to Non-POI: ", not_poi["from_poi_to_this_person"].astype(float).mean(), "\n"

#Average Emails to POI
print "Average Emails to POI from POI: ", poi["from_this_person_to_poi"].astype(float).mean()
#Average Emails to POI(Non-POIs)
print "Average Emails to POI from Non-POI: ", not_poi["from_this_person_to_poi"].astype(float).mean(), "\n"

#Average Shared Receipt with POI
print "Average Shared Receipt with POI (POI): ", poi["shared_receipt_with_poi"].astype(float).mean()
#Average Shared Receipt with POI (Non-POIs)
print "Average Shared Receipt with POI (Non-POIs): ", not_poi["shared_receipt_with_poi"].astype(float).mean()

```

    Average POI Salary:  383444.8823529412
    Average Non-POI Salary:  601152.5 
    
    Average POI Bonus:  2074999.9375
    Average Non-POI Bonus:  2446776.3484848486 
    
    Average POI total payments:  7913589.777777778
    Average Non-POI total payments:  4605104.626168224 
    
    Average POI Total Stock Value:  9165670.944444444
    Average Non-POI Total Stock Value:  6375338.537037037 
    
    Average Emails From POI to POI:  97.78571428571429
    Average Emails From POI to Non-POI:  58.5 
    
    Average Emails to POI from POI:  66.71428571428571
    Average Emails to POI from Non-POI:  36.27777777777778 
    
    Average Shared Receipt with POI (POI):  1783.0
    Average Shared Receipt with POI (Non-POIs):  1058.5277777777778


There are some significant differences between POIs and Non-POIs, specificallly in attributes such as total payments, stock value, email from POI, emails to POI, and emails shared with POIs. Surprisingly, the average salary and bonus are actually larger for Non-POIs.

## Indentifying Outliers


```python
#Determining 99th quantile of salaries
q = df["salary"].astype(float).quantile(0.99)
salary_outliers = df[df["salary"] > q]
salary_outliers = salary_outliers[salary_outliers['salary'] != 'NaN']
salary_outliers
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
      <th>bonus</th>
      <th>deferral_payments</th>
      <th>deferred_income</th>
      <th>director_fees</th>
      <th>email_address</th>
      <th>exercised_stock_options</th>
      <th>expenses</th>
      <th>from_messages</th>
      <th>from_poi_to_this_person</th>
      <th>from_this_person_to_poi</th>
      <th>...</th>
      <th>long_term_incentive</th>
      <th>other</th>
      <th>poi</th>
      <th>restricted_stock</th>
      <th>restricted_stock_deferred</th>
      <th>salary</th>
      <th>shared_receipt_with_poi</th>
      <th>to_messages</th>
      <th>total_payments</th>
      <th>total_stock_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TOTAL</th>
      <td>97343619</td>
      <td>32083396</td>
      <td>-27992891</td>
      <td>1398517</td>
      <td>NaN</td>
      <td>311764000</td>
      <td>5235198</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>48521928</td>
      <td>42667589</td>
      <td>False</td>
      <td>130322299</td>
      <td>-7576788</td>
      <td>26704229</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>309886585</td>
      <td>434509511</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>



Clearly this entry is an anomaly in our dataset and an outlier. This is most likely just a issue with the formatting on the dataset we loaded.


```python
#Removing TOTAL entry from dataframe
salary_outliers = salary_outliers.drop(['TOTAL'])
salary_outliers
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
      <th>bonus</th>
      <th>deferral_payments</th>
      <th>deferred_income</th>
      <th>director_fees</th>
      <th>email_address</th>
      <th>exercised_stock_options</th>
      <th>expenses</th>
      <th>from_messages</th>
      <th>from_poi_to_this_person</th>
      <th>from_this_person_to_poi</th>
      <th>...</th>
      <th>long_term_incentive</th>
      <th>other</th>
      <th>poi</th>
      <th>restricted_stock</th>
      <th>restricted_stock_deferred</th>
      <th>salary</th>
      <th>shared_receipt_with_poi</th>
      <th>to_messages</th>
      <th>total_payments</th>
      <th>total_stock_value</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 21 columns</p>
</div>




```python
df.drop(['TOTAL'],inplace=True)
```

## Additional Features

There may be some ambiguity in the email features. For example, the total number of emails to, from, and shared with POIs might not be the best indicator for those particular metrics, but rather a more descriptive metric may be a ratio of the total emails sent, recieved, and shared to the total emails sent to POIs, recieved from POIs, and shared with POIs.


```python
#Ratio of Emails from POI
df['from_poi_ratio'] = df['from_poi_to_this_person'].astype(float) / (df['from_poi_to_this_person'].astype(float) \
                                                                      + df['from_messages'].astype(float))

#Ratio of Emails to POI
df['to_poi_ratio'] = df['from_this_person_to_poi'].astype(float) / (df['from_this_person_to_poi'].astype(float) \
                                                                    + df['to_messages'].astype(float))

#Ratio of Shared Emails with POI
df['shared_poi_ratio'] =df['shared_receipt_with_poi'].astype(float) / (df['shared_receipt_with_poi'].astype(float) \
                                                                       + df['from_messages'].astype(float) +  \
                                                                       df['from_poi_to_this_person'].astype(float))

```

Here are all the features in the dataset at this point.


```python
#Making list of all features in dataframe
total_features_list = df.columns.values

#Printing List
print total_features_list
```

    ['bonus' 'deferral_payments' 'deferred_income' 'director_fees'
     'email_address' 'exercised_stock_options' 'expenses' 'from_messages'
     'from_poi_to_this_person' 'from_this_person_to_poi' 'loan_advances'
     'long_term_incentive' 'other' 'poi' 'restricted_stock'
     'restricted_stock_deferred' 'salary' 'shared_receipt_with_poi'
     'to_messages' 'total_payments' 'total_stock_value' 'from_poi_ratio'
     'to_poi_ratio' 'shared_poi_ratio']


## Building Dataset and Feature List


```python
# Creating a dictionary from the dataframe
my_dataset = df.to_dict('index')

```

I am going to be using the email ratios and total stock value in my feature list for building my machine learning model.


```python
#Creating my feature list

my_features_list = ['poi', 'total_stock_value', 'total_stock_value','to_poi_ratio','shared_poi_ratio']
```

    ['poi', 'total_stock_value', 'total_stock_value', 'to_poi_ratio', 'shared_poi_ratio']


## Creating Lables and Features for Models


```python
# Extracting features and labels from dataset for local testing

data = featureFormat(my_dataset, my_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

```

## Building and Testing Classifiers


```python
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi', 'salary']






### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
```
