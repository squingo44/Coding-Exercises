# My solutions to the exercises from https://www.machinelearningplus.com/python/101-pandas-exercises-python/

# Alternative solutions are provided in comments

import numpy as np

import pandas as pd

exercise = 1

# Exercise 1: Import pandas and check the version

if exercise == 1:
    print(pd.__version__)

# Exercise 2: Create a pandas series from each of the items provided: a list, array and dictionary

if exercise == 2:
    mylist = list('abcedfghijklmnopqrstuvwxyz')
    myarr = np.arange(26)
    mydict = dict(zip(mylist, myarr))

    list = pd.Series(mylist)
    arr = pd.Series(myarr)
    dict = pd.Series(mydict)

    print(list, arr, dict)

# Exercise 3: Convert the series ser into a dataframe with its index as another column on the dataframe

if exercise == 3:
    mylist = list('abcedfghijklmnopqrstuvwxyz')
    myarr = np.arange(26)
    mydict = dict(zip(mylist, myarr))
    ser = pd.Series(mydict)

    df = ser.to_frame().reset_index()

    print(df.head())

# Exercise 4: Combine ser1 and ser2 to form a dataframe

if exercise == 4:
    ser1 = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))
    ser2 = pd.Series(np.arange(26))

    df = pd.DataFrame({0 : ser1, 1 : ser2})
    
    # df = pd.concat([ser1, ser2], axis=1)

    print(df.head())

# Exercise 5: Give a name to the series ser calling it "alphabets"

if exercise == 5:
    ser = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))

    ser.name = "alphabets"

    print(ser.head())

# Exercise 6: From ser1 remove items present in ser2

if exercise == 6:
    ser1 = pd.Series([1, 2, 3, 4, 5])
    ser2 = pd.Series([4, 5, 6, 7, 8])

    print(ser1[~ser1.isin(ser2)])

# Exercise 7: Get all items of ser1 and ser2 not common to both

if exercise == 7:
    ser1 = pd.Series([1, 2, 3, 4, 5])
    ser2 = pd.Series([4, 5, 6, 7, 8])

    ser_u = pd.Series(np.union1d(ser1, ser2))
    ser_i = pd.Series(np.intersect1d(ser1, ser2))

    print(ser_u[~ser_u.isin(ser_i)])

# Exercise 8: Compute the minimum, lower quartile, median, upper quartile, and maximum of ser

if exercise == 8:
    ser = pd.Series(np.random.normal(10, 5, 25))

    print(np.percentile(ser, q=[0, 25, 50, 75, 100]))

    # print(ser.describe())

# Exercise 9: Calculate the frequency counts of each unique value in ser

if exercise == 9:
    ser = pd.Series(np.take(list("abcdefgh"), np.random.randint(8, size=30)))

    print(ser.value_counts())

# Exercise 10: From ser, keep the top 2 most frequent items as it is and replace everything else as "other"

if exercise == 10:
    ser = pd.Series(np.random.randint(1, 5, 12))

    ser[~ser.isin(ser.value_counts().index[:2])] = "other"

    print(ser)

# Exercise 11: Bin the series ser into 10 equal deciles and replace the values with the bin name

if exercise == 11:
    ser = pd.Series(np.random.random(20))

    print(pd.qcut(ser, 10, labels=["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"]))

# Exercise 12: Reshape the series ser into a dataframe with 7 rows and 5 columns

if exercise == 12:
    ser = pd.Series(np.random.randint(1, 10, 35))

    print(pd.DataFrame(ser.values.reshape((7,5))))

# Exercise 13: Find the positions of elements of ser that are multiples of 3

if exercise == 13:
    ser = pd.Series(np.random.randint(1, 10, 7))
    
    print(ser.index[ser % 3 == 0])

# Exercise 14: From ser extract the items in list pos

if exercise == 14:
    ser = pd.Series(list('abcdefghijklmnopqrstuvwxyz'))
    pos = [0, 4, 8, 14, 20]

    print(ser[pos])

    # print(ser.take(pos)) 

# Exercise 15: Stack ser1 and ser2 vertically and horizontally

if exercise == 15:
    ser1 = pd.Series(range(5))
    ser2 = pd.Series(list('abcde'))

    print(ser1.append(ser2))

    print(pd.concat((ser1, ser2), axis=1))

# Exercise 16: Get the positions of ser2 in ser1 as a list

if exercise == 16:
    ser1 = pd.Series([10, 9, 6, 5, 3, 1, 12, 8, 13])
    ser2 = pd.Series([1, 3, 10, 13])

    print(ser1.index[ser1.isin(ser2)]) # This swaps the results for items 1 and 10?

    # print([np.where(i == ser1)[0].tolist()[0] for i in ser2])

    # print([pd.Index(ser1).get_loc(i) for i in ser2])

# Exercise 17: Compute the mean squared error of truth and pred series

if exercise == 17:
    truth = pd.Series(range(10))
    pred = pd.Series(range(10)) + np.random.random(10)

    MSE =  pred.subtract(truth)**2/truth.size

    print(MSE.sum())

    # print(np.mean((truth - pred)**2))

# Exercise 18: Change the first character of each word to upper case in each word of ser

if exercise == 18:
    ser = pd.Series(['how', 'to', 'kick', 'ass?'])

    print(ser.str.capitalize())

# Exercise 19: Calculate the number of characters in each word in a series

if exercise == 19:
    ser = pd.Series(['how', 'to', 'kick', 'ass?'])

    print(ser.str.len())

# Exercise 20: Find the differences and differences of differences between consecutive elements of ser

if exercise == 20:
    ser = pd.Series([1, 3, 6, 10, 15, 21, 27, 35])
    
    diff = ser.diff()
    print(diff)
    print(diff.diff())

    # print(ser.diff().tolist())
    # print(ser.diff().diff().tolist())

# Exercise 21: Convert a series of date-strings to a timeseries

if exercise == 21:
    ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])

    print(pd.to_datetime(ser))

# Exercise 22: Get the day of month, week number, day of year and day of week from ser

if exercise == 22:
    ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])

    from dateutil.parser import parse
    ser_ts = ser.map(lambda x: parse(x))

    print("Date of month: ", ser_ts.dt.day.tolist())

    print("Week number: ", ser_ts.dt.isocalendar().week.tolist())

    print("Day number of year: ", ser_ts.dt.dayofyear.tolist())

    print("Day of week: ", ser_ts.dt.day_name().tolist())

# Exercise 23: Change ser to dates that start with the 4th of the respective months

if exercise == 23:
    ser = pd.Series(['Jan 2010', 'Feb 2011', 'Mar 2012'])

    from dateutil.parser import parse
    ser_ts = ser.map(lambda x: parse(x))

    print(ser_ts.apply(lambda dt: dt.replace(day=4)))

# Exercise 24: From ser extract words that contain at least two vowels

if exercise == 24:
    ser = pd.Series(["Apple", "Orange", "Plan", "Python", "Money"])

    print(ser.where(ser.str.contains("([aeiouAEIOU]).*[aeiouAEIOU]")).dropna())

    # from collections import Counter
    # mask = ser.map(lambda x: sum([Counter(x.lower()).get(i,0) for i in list("aeiou")]) >= 2)
    # print(ser.mask)

# Exercise 25: Extract the valid emails from the series emails

if exercise == 25:
    emails = pd.Series(['buying books at amazom.com', 'rameses@egypt.com', 'matt@t.co', 'narendra@modi.com'])
    pattern ='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,4}'

    print(emails.where(emails.str.contains(pattern)).dropna())

    # solution as a series of strings:

    # import re
    # mask = emails.map(lambda x: bool(re.match(pattern, x)))
    # print(emails[mask])

    # solution as a series of lists:

    # emails.str.findall(pattern, flags=re.IGNORECASE)

    # solution as a list:

    # [x[0] for x in [re.findall(pattern, email) for email in emails] if len(x) > 0]

# Exercise 26: Compute the mean of weights of each fruit.

if exercise == 26:
    fruit = pd.Series(np.random.choice(['apple', 'banana', 'carrot'], 10))
    weights = pd.Series(np.linspace(1, 10, 10))

    df = pd.DataFrame([fruit, weights]).T

    print(df.pivot(columns=0, values=1).mean())

    # weights.groupby(fruit).mean()

# Exercise 27: Compute the Euclidean distance between the series of points p and q without using a packaged formula

if exercise == 27:
    p = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    q = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

    print(sum((p-q)**2)**(0.5))

    # np.linalg.norm(p-q)

# Exercise 28: Get the positions of peaks in ser

if exercise == 28:
    ser = pd.Series([2, 10, 3, 4, 9, 10, 2, 7, 3])

    print(np.where(np.diff(np.sign(np.diff(ser))) == -2)[0] + 1)

# Exercise 29: Replace the spaces in my_str with the least frequent character

if exercise == 29:
    my_str = 'dbc deb abed gade'

    str = pd.Series(my_str[i] for i in range(len(my_str))).T
    print("".join(str.replace(to_replace = " ", value = str.value_counts().idxmin())))

    # ser = pd.Series(list('dbc deb abed gade'))
    # freq = ser.value_counts()
    # print(freq)
    # least_freq = freq.dropna().index[-1]
    # "".join(ser.replace(' ', least_freq))

# Exercise 30: Create a TimeSeries starting "2000-01-01" and 10 Saturdays and have random numbers as values in another column

if exercise == 30:
    print(pd.Series(np.random.randint(1,10,10), pd.date_range('2000-01-01', periods=10, freq='W-SAT')))

# Exercise 31: Make all missing dates appear and fill up with value from previous date

if exercise == 31:
    ser = pd.Series([1,10,3,np.nan], index=pd.to_datetime(['2000-01-01', '2000-01-03', '2000-01-06', '2000-01-08']))
    
    print(ser.resample('D').ffill())

# Exercise 32: Compute autocorrelations for the first 10 lags of ser and find out which lag has the largest correlation

if exercise == 32:
    ser = pd.Series(np.arange(20) + np.random.normal(1, 10, 20))

    autocorrelations = np.array([ser.autocorr(i).round(2) for i in range(1,11)])
    print(autocorrelations)
    print("Lag having highest correlation: ", np.argmax(autocorrelations)+1)

# Exercise 33: Import every 50th row of the Boston Housing dataset as a dataframe

if exercise == 33:
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    out = pd.read_csv(url).iloc[::50, :]
    print(out)

# Exercise 34: Import the Boston Housing dataset, but while importing change the "medv" column so that values < 25 become "Low" and values >= 25 become "High"

if exercise == 34:
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    housingdata = pd.read_csv(url,converters = {"medv": lambda x: "High" if float(x) > 25 else "Low"})
    print(housingdata)

    # import csv
    # with open("BostonHousing.csv", "r") as f:
    #   reader = csv.reader(f)
    #   out = []
    #   for i, row in enumerate(reader):
    #       if i > 0:
    #           row[13] = "High" if float(row[13]) > 25 else "Low"
    #       out.append(row)
    # df = pd.DataFrame(out[1:], columns=out[0])
    # print(df.head)

# Exercise 35: Create a dataframe with rows as strides from a given series

if exercise == 35:
    L = pd.Series(range(15))

    def gen_strides(a, stride_len=5, window_len=5):
        n_strides = ((a.size - window_len)//stride_len) + 1
        return np.array([a[s:(s + window_len)] for s in np.arange(0, a.size, stride_len)[:n_strides]])
    print(gen_strides(L, stride_len = 2, window_len = 4))
    
# Exercise 36: Import "crim" and "medv" columns of the Boston Housing dataset as a dataframe

if exercise == 36:
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    out = pd.read_csv(url).iloc[:, [0,13]]
    print(out)

    # out = pd.read_csv(url, usecols=["crim", "medv"])

# Exercise 37: Get the number of rows, columns, datatype and summary statistics of each column  of the cars93 dataset; also get its numpy and list equivalent

if exercise == 37:
    url = "https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv"
    out = pd.read_csv(url)
    print(out.shape)
    print(out.dtypes)
    print(out.dtypes.value_counts())
    print(out.describe())
    print(out.values)
    print(out.values.tolist())

# Exercise 38: Extract the row and column number of a cell meeting a specific criterion

if exercise == 38:
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
    df.loc[df.Price == np.max(df.Price), ['Manufacturer', 'Model', 'Type']]
    row, col = np.where(df.values == np.max(df.Price))
    print("Row: " + str(row) + "; Col: " + str(col))
    print(df.iat[row[0], col[0]])

# Exercise 39: Rename the column Type as CarType in df and replace the "." in column names with "_"

if exercise == 39:
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

    df = df.rename(columns = {"Type":"CarType"})
    df.columns = df.columns.map(lambda x: x.replace(".", "_"))
    print(df.columns)

# Exercise 40: Check if df has any missing values

if exercise == 40:
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

    print(df.isnull().values.any())

# Exercise 41: Count the number of missing values in each column of df. Which column has the maximum number of missing values?

if exercise == 41:
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

    print(df.isna().sum())
    print(df.columns[df.isna().sum().argmax()])

# Exercise 42: Replace missing values in Min.Price and Max.Price with their respective mean

if exercise == 42:
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

    df[["Min.Price", "Max.Price"]] = df[["Min.Price", "Max.Price"]].apply(lambda x: x.fillna(x.mean()))
    print(df)
    
# Exercise 43: Use apply to replace the missing values n Min.Price with the column's mean and those in Max.Price with the column's median

if exercise == 43:
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

    df[["Min.Price"]] = df[["Min.Price"]].apply(lambda x: x.fillna(x.mean()))
    df[["Max.Price"]] = df[["Max.Price"]].apply(lambda x: x.fillna(x.median()))
    print(df.head())

    # d = {'Min.Price': np.nanmean, 'Max.Price': np.nanmedian}
    # df[['Min.Price', 'Max.Price']] = df[['Min.Price', 'Max.Price']].apply(lambda x, d: x.fillna(d[x.name](x)), args=(d, ))

# Exercise 44: Get the first column (a) in df as a dataframe

if exercise == 44:
    df = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde'))

    print(df[["a"]])

# Exercise 45: In df, interchange columns "a" and "c"; create a generic function to interchange two columns without hardcoded names; Sort the columns in reverse alphabetical order

if exercise == 45:
    df = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde'))

    print(df.reindex(columns=["c","b","a","d","e"]))
    # print(df[list("cbade")])
    
    def switch_columns(df, col1=None, col2=None):
        colnames = df.columns.tolist()
        i1, i2 = colnames.index(col1), colnames.index(col2)
        colnames[i2], colnames[i1] = colnames [i1], colnames[i2]
        return df[colnames]
    
    print(switch_columns(df, "a", "c"))

    print(df.sort_index(axis=1, ascending=False))
    # print(df[sorted(df.columns, reverse=True)])

# Exercise 46: Change the pandas display settings on printing df so that it shows a maximum of 10 rows and 10 columns

if exercise == 46:
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

    print(df.iloc[:10,:10])

    # pd.set_option("display.max_columns", 10)
    # pd.set_option("display.max_rows", 10)

# Exercise 47: Suppress scientific notation like "e-03" in df and print up to 4 numbers after decimal

if exercise == 47:
    df = pd.DataFrame(np.random.random(4)**10, columns=['random'])

    pd.set_option("display.float_format", '{:.4f}'.format)
    print(df)

    # print(df.round(4))

    # df.applymap(lambda x: '%.4f' % x)

# Exercise 48: Format the values in column "random" of df as percentages

if exercise == 48:
    df = pd.DataFrame(np.random.random(4), columns=['random'])

    out = df.style.format({'random': '{0:.2%}'.format})
    print(out)

# Exercise 49: From df, filter the "Manufacturer", "Model", and "Type" for every 20th row starting from row 0

if exercise == 49:
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

    print(df.iloc[::20, :][["Manufacturer", "Model", "Type"]])

# Exercise 50: In df, replace NaNs with "missing" in columns "Manufacturer", "Model", and "Type" and create an index as a combination of these three columns and check if the index is a primary key

if exercise == 50:
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv', usecols=[0,1,2,3,5])

    df[["Manufacturer", "Model", "Type"]] = df[["Manufacturer", "Model", "Type"]].fillna("missing")
    df.index = df.Manufacturer + "_" + df.Model + "_" + df.Type
    print(df)
    print(df.index.is_unique) # checks if every element of the new index is unique

# Exercise 51: Find the row position of the 5th largest value of column "a" in df

if exercise == 51:
    df = pd.DataFrame(np.random.randint(1, 30, 30).reshape(10,-1), columns=list('abc'))

    print(df["a"].argsort()[::-1][5])

# Exercise 52: In ser, find the position of the 2nd largest value greater than the mean

if exercise == 52:
    ser = pd.Series(np.random.randint(1, 100, 15))

    print(ser)
    print(ser.where(ser > ser.mean()).dropna().sort_values(ascending=False).index[1])

    # print(np.argwhere(ser > ser.mean())[1])

# Exercise 53: Get the last two rows of df whose row sum is greater than 100

if exercise == 53:
    df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))

    print(df.where(df.sum(axis=1) > 100).dropna().tail(2))

    # rowsums = df.apply(np.sum, axis=1)
    # last_two_rows = df.iloc[np.where(rowsums > 100)[0][-2:], :]

# Exercise 54: Replace all values of ser below the 5th percentile and above the 95th percentile with the respective 5th percentile and 95th percentile values

if exercise == 54:
    ser = pd.Series(np.logspace(-2, 2, 30))

    def cap_outliers(ser, low_perc, high_perc):
        if low_perc > high_perc:
            low_perc, high_perc = high_perc, low_perc
        if low_perc > 1 and high_perc > 1:
            low_perc, high_perc = low_perc/100, high_perc/100
        low, high = ser.quantile([low_perc, high_perc])
        ser[ser < low] = low
        ser[ser > high] = high
        return(ser)
    
    print(cap_outliers(ser, 5, 95))

# Exercise 55: Reshape df to the largest possible square with negative values removed. Drop the smallest values if necessary. The order of the positive values in the result should remain the same as the original

if exercise == 55:
    df = pd.DataFrame(np.random.randint(-20, 50, 100).reshape(10,-1))

    print(df)
    df = df.where(df >= 0).to_numpy().flatten()
    df = df[~np.isnan(df)]
    a = int(np.size(df)**0.5)
    b = np.size(df) - a**2
    print("Number of positive elements:", np.size(df), "\nOutput array size:", a, "\nNumber of elements to remove:", b)
    df = df[sorted(np.argsort(df)[b:])]
    df = df.reshape(a,a)
    df = pd.DataFrame(df, dtype="int64")
    print(df)

    # arr = df[df > 0].values.flatten()
    # arr_qualified = arr[~np.isnan(arr)]

    # n = int(np.floor(arr_qualified.shape[0]**.5))

    # top_indexes = np.argsort(arr_qualified)[::-1]
    # output = np.take(arr_qualified, sorted(top_indexes[:n**2])).reshape(n,-1)
    # print(output)

# Exercise 56: Swap rows 1 and 2 in df

if exercise == 56:
    df = pd.DataFrame(np.arange(25).reshape(5, -1))

    print(df)
    def swap_rows(df, i1, i2):
        b, c = df.iloc[i1], df.iloc[i2]
        temp = df.iloc[i1].copy()
        df.iloc[i1] = c
        df.iloc[i2] = temp
        return(df)
    print(swap_rows(df, 1, 2))

# Exercise 57: Reverse all rows of the dataframe df

if exercise == 57:
    df = pd.DataFrame(np.arange(25).reshape(5, -1))

    print(df.iloc[::-1])

# Exercise 58: Get one-hot encodings for column "a" in df and append it as columns

if exercise == 58:
    df = pd.DataFrame(np.arange(25).reshape(5,-1), columns=list('abcde'))

    print(pd.concat([pd.get_dummies(df['a']), df.loc[:, df.columns!="a"]], axis=1))

# Exercise 59: Obtain the column name with the highest number of row-wise maximums in df

if exercise == 59:
    df = pd.DataFrame(np.random.randint(1,100, 40).reshape(10, -1))

    print(df)
    print(df.apply(np.argmax, axis=1).value_counts().index[0])

# Exercise 60: Create a new column such that each row contains the row number of nearest row-record by Euclidean distance

if exercise == 60:
    df = pd.DataFrame(np.random.randint(1,100, 40).reshape(10, -1), columns=list('pqrs'), index=list('abcdefghij'))

    distances = pd.DataFrame(1000, index=df.index, columns=df.index)
    for i, m in df.iterrows():
        for j, n in df.iterrows():
            if i != j:
                distances.loc[i,j] = np.linalg.norm(m-n)
    l = (list(distances))
    l2 = distances.apply(np.argmin, axis=0).tolist()
    out = []
    for i in l2:
        out.append(l[i])
    df["nearest_value"] = out
    df["dist"] = distances.apply(np.min, axis=0)
    print(df)

    # nearest_rows = []
    # nearest_distance = []

    # for i, row in df.iterrows():
    #    curr = row
    #    rest = df.drop(i)
    #    e_dists = {} # init dict to store Euclidean dists for current row
    #    for j, contestant in rest.iterrows():
    #        e_dists.update({j: round(np.linalg.norm(curr.values - contestant.values))})
    #    nearest_rows.append(max(e_dists, key=e_dists.get))
    #    nearest_distance.append(max(e_dists.values()))

    # df["nearest_row"] = nearest_rows
    # df["dist"] = nearest_distance
    
# Exercise 61: Compute the maximum possible absolute correlation value of each column against other columns in df

if exercise == 61:
    df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1), columns=list('pqrstuvwxy'), index=list('abcdefgh'))

    print("df:\n", df)
    correlations = pd.DataFrame(0, index=df.columns, columns=df.columns)

    for i, m in df.iteritems():
        for j, n in df.iteritems():
            correlations.loc[i,j] = abs(m.corr(n))
    print("\nColumn correlations:\n", np.round(correlations, 2))

    # abs_corrmat = np.abs(df.corr())

    max_corr = correlations.apply(lambda x: sorted(x)[-2])
    print("Maximum correlation possible for each column: ", np.round(max_corr.tolist(), 2))

# Exercise 62: Compute the minimum-by-maximum for every row of df

if exercise == 62:
    df = pd.DataFrame(np.random.randint(1,100,80).reshape(8,-1))
    
    out = df.min()/df.max()
    print(round(out,2))

# Exercise 63: Create a new column "penultimate" which has the second largest value of each row of df

if exercise == 63:
    df = pd.DataFrame(np.random.randint(1,100,80).reshape(8, -1))

    df["penultimate"] = df.apply(lambda x: x.nlargest(2).iloc[1], axis=1)

    # df["penultimate"] = df.apply(lambda x: x.sort_values().unique()[-2], axis=1)

    print(df)

# Exercise 64: Normalise all columns of df by subtracting the column mean and divide by standard deviation; range all columns of df such that the minimum value in each column is 0 and max is 1

if exercise == 64:
    df = pd.DataFrame(np.random.randint(1,100,80).reshape(8,-1))

    df_min_max_scaled = df.copy()
    df_min_max_scaled = (df_min_max_scaled - df_min_max_scaled.min()) / (df_min_max_scaled.max() - df_min_max_scaled.min())  

    print(df_min_max_scaled)

    # out1 = df.apply(lambda x: ((x - x.mean())/x.std()).round(2))
    # print(out1)

    # out2 = df.apply(lambda x: ((x.max() - x)/(x.max() - x.min())).round(2))
    # print(out2)

# Exercise 65: Compute the correlation of each row of df with its succeeding row

if exercise == 65:
    df = pd.DataFrame(np.random.randint(1,100,80).reshape(8,-1))

    print([df.iloc[i].corr(df.iloc[i+1]).round(2) for i in range(df.shape[0])[:-1]])

# Exercise 66: Replace values in both diagonals of df with 0

if exercise == 66:
    df = pd.DataFrame(np.random.randint(1,100, 100).reshape(10, -1))

    for i, row in df.iterrows():
        df.iloc[i,i] = 0
        df.iloc[-1-i,i] = 0

    print(df)

# Exercise 67: From df_grouped, get the group beloning to "apple" as a dataframe

if exercise == 67:
    df = pd.DataFrame({'col1': ['apple', 'banana', 'orange'] * 3,
                   'col2': np.random.rand(9),
                   'col3': np.random.randint(0, 15, 9)})

    df_grouped = df.groupby(['col1'])

    print(df_grouped.get_group("apple"))

# Exercise 68: In df, find the second largest value of "taste" for "banana"

if exercise == 68:
    df = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,
                    'rating': np.random.rand(9),
                    'price': np.random.randint(0, 15, 9)})

    print(df.groupby(["fruit"]).get_group("banana").sort_values(["rating"]).iloc[-2])

# Exercise 69: In df, compute the mean price of every fruit, while keeping the fruit as another column instead of index

if exercise == 69: 
    df = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,
                    'rating': np.random.rand(9),
                    'price': np.random.randint(0, 15, 9)})

    out = df.groupby("fruit", as_index=False)["price"].mean()
    print(df)
    print(out)

# Exercise 70: Join dataframes df1 and df2 by "fruit-pazham" and "weight-kilo"

if exercise == 70:
    df1 = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,
                        'weight': ['high', 'medium', 'low'] * 3,
                        'price': np.random.randint(0, 15, 9)})

    df2 = pd.DataFrame({'pazham': ['apple', 'orange', 'pine'] * 2,
                        'kilo': ['high', 'low'] * 3,
                        'price': np.random.randint(0, 15, 6)})

    print(pd.merge(df1, df2, how='inner', left_on=['fruit', 'weight'], right_on=['pazham', 'kilo'], suffixes=['_left', '_right']))

# Exercise 71: From df1, remove the rows that are present in df2

if exercise == 71:
    df1 = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,
                        'weight': ['high', 'medium', 'low'] * 3,
                        'price': np.random.randint(0, 15, 9)})

    df2 = pd.DataFrame({'pazham': ['apple', 'orange', 'pine'] * 2,
                        'kilo': ['high', 'low'] * 3,
                        'price': np.random.randint(0, 15, 6)})

    print(df1[~df1.isin(df2).all(1)])

# Exercise 72: Get the positions of rows where values of two columns match

if exercise == 72:
    df = pd.DataFrame({'fruit1': np.random.choice(['apple', 'orange', 'banana'], 10),
                        'fruit2': np.random.choice(['apple', 'orange', 'banana'], 10)})

    print(df)

    print(np.where(df.fruit1 == df.fruit2))

# Exercise 73: Create two new columns in df, one of which is "lag1" (shift column "a" down by 1 row) of column "a" and the other is "lead1"(shift column "b" up by 1 row)

if exercise == 73:
    df = pd.DataFrame(np.random.randint(1, 100, 20).reshape(-1, 4), columns = list('abcd'))

    df['a_lag1'] = df['a'].shift(1)
    df['b_lead1'] = df['b'].shift(-1)
    print(df)

# Exercise 74: Get the frequency of unique values in the entire dataframe df

if exercise == 74:
    df = pd.DataFrame(np.random.randint(1,10,20).reshape(-1,4), columns=list("abcd"))

    print(df)

    print(pd.value_counts(df.values.ravel()))

# Exercise 75: Split the string column in df to form a dataframe with 3 columns

if exercise == 75:
    df = pd.DataFrame(["STD, City    State",
    "33, Kolkata    West Bengal",
    "44, Chennai    Tamil Nadu",
    "40, Hyderabad    Telengana",
    "80, Bangalore    Karnataka"], columns=['row'])

    print(df)

    df_out = df.row.str.split(',|\t', expand=True)
    new_header = df_out.iloc[0]
    df_out = df_out[1:]
    df_out.columns = new_header

    print(df_out)

# Exercises 76-101 were never published