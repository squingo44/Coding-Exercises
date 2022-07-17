# My solutions to the exercises from https://www.machinelearningplus.com/python/101-numpy-exercises-python/

# Alternative solutions are provided in comments

import numpy as np

exercise = 1

# Exercise 1: Import numpy as np and print the version number

if exercise == 1:
    print(np.__version__)

# Exercise 2: Create 1D array of numbers from 0 to 9

if exercise == 2:
    arr = np.arange(10)
    print(arr)

# Exercise 3: Create a 3x3 numpy array of all Trues

if exercise == 3:
    arr = np.full((3,3),True)
    print(arr)

# Exercise 4: Extract all odd numbers from the input array

if exercise == 4:
    arr = np.arange(10)
    out = np.array(arr[arr % 2 == 1])
    print(out)

# Exercise 5: Replace all odd numbers in the input array with -1

if exercise == 5:
    arr = np.arange(10)
    arr[arr % 2 == 1] = -1
    print(arr)

# Exercise 6: Replace all odd numbers in the input array with -1, without changing the input array

if exercise == 6:
    arr = np.arange(10)
    out = np.where(arr % 2 == 1, -1, arr)
    print(arr)
    print(out)

# Exercise 7: Convert a 1D array into a 2D array with two rows

if exercise == 7:
    arr = np.arange(10)
    arr = np.reshape(arr, (2, -1))
    print(arr)

# Exercise 8: Stack two input arrays vertically

if exercise == 8:
    a = np.arange(10).reshape(2,-1)
    b = np.repeat(1, 10).reshape(2,-1)
    out = np.concatenate((a, b), axis = 0)
    print(out)

# Exercise 9: Stack two input arrays horizontally

if exercise == 9:
    a = np.arange(10).reshape(2,-1)
    b = np.repeat(1, 10).reshape(2,-1)
    out = np.concatenate((a, b), axis = 1)
    print(out)

# Exercise 10: Create the following pattern without hardcoding, use only numpy functions and input array

if exercise == 10:
    a = np.array([1,2,3])
    b = np.repeat(a, 3)
    out = np.concatenate((b, a, a, a), axis = 0)


    # out = np.r_[np.repeat(a, 3), np.tile(a, 3) is quicker

# Exercise 11: Get the common items between two input arrays

if exercise == 11:
    a = np.array([1,2,3,2,3,4,3,4,5,6])
    b = np.array([7,2,10,2,7,4,9,4,9,8])
    out = np.intersect1d(a,b)
    print(out)

# Exercise 12: From array a, remove all items present in array b

if exercise == 12:
    a = np.array([1,2,3,4,5])
    b = np.array([5,6,7,8,9])
    out = np.setdiff1d(a,b)
    print(out)

# Exercise 13: Get the positions where elements of a and b match

if exercise == 13:
    a = np.array([1,2,3,2,3,4,3,4,5,6])
    b = np.array([7,2,10,2,7,4,9,4,9,8])
    out = np.where(a==b)
    print(out)

# Exercise 14: Get all items between 5 and 10 from an input array

if exercise == 14:
    a = np.array([2, 6, 1, 9, 10, 3, 27])
    out = np.array(a[(a >= 5) & (a <= 10)])
    print(out)

# Exercise 15: Convert the function maxx that works on two scalars, to work on two arrays

if exercise == 15:

    def maxx(x, y):
        if x >= y:
            return x
        else:
            return y

    a = np.array([5, 7, 9, 8, 6, 4, 5])
    b = np.array([6, 3, 4, 8, 9, 7, 1])

    pair_max = np.vectorize(maxx, otypes = [float])

    print(pair_max(a,b))

# Exercise 16: Swap columns 1 and 2 in the input array

if exercise == 16:
    arr = np.arange(9).reshape(3,3)
    arr = arr[:, [1,0,2]]
    print(arr)

# Exercise 17: Swap rows 1 and 2 in the input array

if exercise == 17:
    arr = np.arange(9).reshape(3,3)
    arr = arr[[1,0,2], :]
    print(arr)

# Exercise 18: Reverse the rows of a 2D input array

if exercise == 18:
    arr = np.arange(9).reshape(3,3)
    arr = np.flip(arr, axis=0)
    print(arr)

    # arr = arr[::-1] is faster

# Exercise 19: Reverse the columns of a 2D input array

if exercise == 19:
    arr = np.arange(9).reshape(3,3)
    arr = arr[:, ::-1]
    print(arr)

# Exercise 20: Create a 5x3 array of random numbers between 5 and 10

if exercise == 20:
    arr = np.random.uniform(5,10,(5,3))
    print(arr)

# Exercise 21: Show only 3 decimal places of the random input array

if exercise == 21:
    rand_arr = np.random.random((5,3))
    np.set_printoptions(precision=3)
    print(rand_arr)

# Exercise 22: Pretty print a random input array by suppressing scientific notation

if exercise == 22:
    # Create the random array
    np.random.seed(100)
    rand_arr = np.random.random([3,3])/1e3
    np.set_printoptions(suppress=True)
    print(rand_arr)

# Exercise 23: Limit the number of elements of an input array printed to 6

if exercise == 23:
    a = np.arange(15)
    np.set_printoptions(threshold = 6)
    print(a)

# Exercise 24: Print a full input array without truncating

if exercise == 24:
    np.set_printoptions(threshold = 6)
    a = np.arange(15)
    np.set_printoptions(threshold = 1000)
    print(a)

# Exercise 25: Import the iris dataset keeping the text intact

if exercise == 25:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris = np.genfromtxt(url, delimiter=',', dtype='object')
    names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
    print(iris[:4])

# Exercise 26: Extract the text column species from the iris dataset

if exercise == 26:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)
    species = np.array([row[4] for row in iris_1d])
    print(species[:5])

# Exercise 27: Convert the 1D iris to 2D array iris_2d by omitting the species text field

if exercise == 27:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)
    iris_2d = np.array([row.tolist()[:4] for row in iris_1d])
    
    print(iris_2d[:4])

# Exercise 28: Find the mean, median, and standard deviation of iris' sepallength

if exercise == 28:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris = np.genfromtxt(url, delimiter=',', dtype=None)
    iris_2d = np.array([row.tolist()[:4] for row in iris])

    sepallength = np.array([row[0] for row in iris_2d])
    print(np.median(sepallength), np.mean(sepallength), np.std(sepallength))

    # sepallength = np.genfromtxt(url, delimiter=",", dtype="float", usecols=[0]) is faster

# Exercise 29: Create a normalised form of iris' sepallength with values between 0 and 1

if exercise == 29:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])

    normalised_sepallength = (sepallength - np.min(sepallength))/np.ptp(sepallength)
    print(normalised_sepallength[:4], np.min(normalised_sepallength), np.max(normalised_sepallength))

# Exercise 30: Compute the softmax score of sepallength

if exercise == 30:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    print(softmax(sepallength))

# Exercise 31: Find the 5th and 95th percentiles of iris' sepallength

if exercise == 31:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])

    print(np.percentile(sepallength, [5, 95]))

# Exercise 32: Insert np.nan values at 20 random positions in an input array

if exercise == 32:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')

    iris_2d[(np.random.randint(0, np.size(iris_2d, 0), 20), np.random.randint(0, np.size(iris_2d, 1), 20))] = np.nan
    print(iris_2d)

    # i, j = np.where(iris_2d)
    # iris_2d[np.random.choice((i), 20), np.random.choice((j), 20)] = np.nan

# Exercise 33: Find the number and position of missing values in iris_2d's sepallength

if exercise == 33:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris_2d = np.genfromtxt(url, delimiter=',', dtype='float')
    iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

    print("Number of missing values:\n", np.isnan(iris_2d[:,0]).sum())
    print("Position of missing values:\n ", np.where(np.isnan(iris_2d[:,0])))

# Exercise 34: Filter the rows of iris_2d that have petallength > 1.5 and sepallength < 5

if exercise == 34:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

    print(iris_2d[(iris_2d[:,2] > 1.5) & (iris_2d[:,0] < 5)])

# Exercise 35: Select the rows of iris_2d that do not have any nan values

if exercise == 35:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
    iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

    print(iris_2d[np.array([~np.any(np.isnan(row)) for row in iris_2d])][:5])

    # print(iris_2d[np.sum(np.isnan(iris_2d), axis = 1) == 0][:5])

# Exercise 36: Find the correlation between sepallength and petallength in iris_2d

if exercise == 36:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

    print(np.corrcoef(iris_2d[:,0], iris_2d[:,2])[0,1])

# Exercise 37: Determine if iris_2d has any missing values

if exercise == 37:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

    print(np.isnan(iris_2d).any())

# Exercise 38: Replace all values of nan with 0 in iris_2d

if exercise == 38:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
    iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

    iris_2d[np.isnan(iris_2d)] = 0
    print(iris_2d[:5])

# Exercise 39: Find the unique values and the count of the unique values in iris' species

if exercise == 39:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris = np.genfromtxt(url, delimiter=',', dtype='object')
    names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

    print(np.unique(iris[:,4], return_counts=True))

# Exercise 40: Bin the petallength column of iris_2d to form a text array of "small", "medium", and "large"

if exercise == 40:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris = np.genfromtxt(url, delimiter=',', dtype='object')
    names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

    petallength_bin = np.digitize(iris[:,2].astype("float"), [0, 3, 5, 10])
    label_map = {1: "small", 2: "medium", 3: "large", 4: np.nan}
    petallength_cat = [label_map[x] for x in petallength_bin]
    print(petallength_cat)
    iris[:,2] = petallength_cat
    print(iris)

# Exercise 41: Create a new column for volume in iris_2d, where volume = (pi * petallength * sepallength^2)/3

if exercise == 41:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')
    names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

    sepallength = iris_2d[:,0].astype("float")
    petallength = iris_2d[:,2].astype("float")
    volume = (np.pi * petallength * (sepallength**2))/3
    volume = volume[:,np.newaxis]
    print(np.hstack([iris_2d, volume]))

# Exercise 42: Randomly sample iris' species such that setose is twice the number of versicolor and virginica

if exercise == 42:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris = np.genfromtxt(url, delimiter=',', dtype='object')

    species = iris[:,4]

    probs = np.r_[np.linspace(0, 0.5, num = 50), np.linspace(0.501, 0.75, num=50), np.linspace(0.751, 1, num=50)]
    index = np.searchsorted(probs, np.random.random(150))
    species_out = species[index]
    print(np.unique(species_out, return_counts=True))

    # a = np.array(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
    # species_out = np.random.choice(a, 150, p=[0.5, 0.25, 0.25])
    # print(species_out)

# Exercise 43: What is the value of the second largest petallength of species setosa

if exercise == 43:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris = np.genfromtxt(url, delimiter=',', dtype='object')
    names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

    print(np.partition(np.unique(iris[np.where(iris[:,4] == b'Iris-setosa')][:,2]), -2)[-2])

# Exercise 44: Sort the iris dataset based on the sepallength column

if exercise == 44:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris = np.genfromtxt(url, delimiter=',', dtype='object')
    names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

    print(iris[iris[:, 0].argsort()])

# Exercise 45: Find the most frequent value of petallength in the iris dataset

if exercise == 45:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris = np.genfromtxt(url, delimiter=',', dtype='object')
    names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

    v,c = np.unique(iris[:,2], return_counts=True)
    print(v[np.argmax(c)])

# Exercise 46: Find the position of the first occurrence of a value greater than 1.0 in petalwidth

if exercise == 46:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris = np.genfromtxt(url, delimiter=',', dtype='object')

    print(np.argwhere(iris[:, 3].astype(float) > 1.0)[0])

# Exercise 47: From the input array, replace all values greater than 30 to 30 and less than 10 to 10.

if exercise == 47:
    a = np.random.uniform(1,50, 20)

    a[np.where(a > 30)] = 30
    a[np.where(a < 10)] = 10
    print(a)

    # print(np.clip(a. a_min=10, a_max=30)

# Exercise 48: Get the positions of the top 5 values in a input array

if exercise == 48:
    a = np.random.uniform(1,50, 20)

    print(np.argsort(a)[-5:])

# Exercise 49: Compute the counts of unique row-wise values

if exercise == 49:
    arr = np.random.randint(1,11,size=(6, 10))

    out = np.zeros((6,10), dtype=np.int64)

    for i in range(6):
        for j in range(10):
            out[i][j] = np.count_nonzero(arr[i] == j+1)

    print(arr)
    print(out)

    # def counts_of_all_values_rowwise(arr2d):
        # Unique values and its counts row wise
        # num_counts_array = [np.unique(row, return_counts=True) for row in arr2d]

        # Counts of all values row wise
        # return([[int(b[a==i]) if i in a else 0 for i in np.unique(arr2d)] for a, b in num_counts_array])

    # print(np.arange(1,11))
    # counts_of_all_values_rowwise(arr)

# Exercise 50: Convert array_of_arrays into a flat linear 1d array

if exercise == 50:
    arr1 = np.arange(3)
    arr2 = np.arange(3,7)
    arr3 = np.arange(7,10)

    array_of_arrays = np.array([arr1, arr2, arr3])
    
    print(np.array([a for arr in array_of_arrays for a in arr]))

    # print(np.concatenate(array_of_arrays))

# Exercise 51: Compute the one-hot encodings of an input array

if exercise == 51:
    arr = np.random.randint(1,4, size=6)

    out = np.zeros((6,3))
    for i in range(len(arr)):
        out[i, arr[i]-1] = 1
    print(out)

# Exercise 52: Create row numbers grouped by a categorical variable

if exercise == 52:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
    species_small = np.sort(np.random.choice(species, size=20))

    print([i for val in np.unique(species_small) for i, grp in enumerate(species_small[species_small==val])])

# Exercise 53: Create group ids based on a given categorical variable

if exercise == 53:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
    species_small = np.sort(np.random.choice(species, size=20))

    print([np.argwhere(np.unique(species_small) == s).tolist()[0][0] for val in np.unique(species_small) for s in species_small[species_small==val]])

# Exercise 54: Create the ranks for the given input array

if exercise == 54:
    a = np.random.randint(20, size=10)

    t = np.argsort(a)
    ranks = np.empty_like(t)
    ranks[t] = np.arange(len(a))
    print(ranks)

    # print(a.argsort().argsort())

# Exercise 55: Create a rank array of the same shape as given numeric array a

if exercise == 55:
    a = np.random.randint(20, size=[2,5])

    print(a)
    print(a.ravel().argsort().argsort().reshape(a.shape))

# Exercise 56: Compute the maximum for each row in a given array

if exercise == 56:
    a = np.random.randint(1,10, [5,3])

    print(np.amax(a, axis=1))

    # np.apply_along_axis(np.max, arr=a, axis=1)

# Exercise 57: Compute the min-by-max for each row for a given 2d array

if exercise == 57:
    a = np.random.randint(1,10, [5,3])

    print(a)
    print(np.amin(a, axis=1) / np.amax(a, axis=1))

    # np.apply_along_axis(lambda x: np.min(x)/np.max(x), arr=a, axis=1)

# Exercise 58: Find the duplicate entries of the given array and mark them as true, and first time occurrences as false

if exercise == 58:
    a = np.random.randint(0, 5, 10)
    print(a)
    out = np.full(a.shape, True)
    out[np.unique(a, return_index=True)[1]] = False
    print(out)

# Exercise 59: Find the mean of a numeric column grouped by a categorical column in a 2d array

if exercise == 59:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris = np.genfromtxt(url, delimiter=',', dtype='object')
    names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

    numeric_column = iris[:,1].astype("float")
    grouping_column = iris[:,4]
    print([[group_val, numeric_column[grouping_column==group_val].mean()] for group_val in np.unique(grouping_column)])

    # output = []
    # for group_val in np.unique(grouping_colun):
    #    output.append([group_val, numeric_column[grouping_colum ==group_val].mean()])

# Exercise 60: Import the image from the following URL and convert it to a numpy array

if exercise == 60:
    URL = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg'
    
    import PIL, requests
    from PIL import Image
    from io import BytesIO

    response = requests.get(URL)
    I = Image.open(BytesIO(response.content))
    arr = np.asarray(I)
    print(arr)

# Exercise 61: Drop all nan values from a 1d input array

if exercise == 61:
    x = np.array([1,2,3,np.nan,5,6,7,np.nan])

    print(x[~np.isnan(x)])

# Exercise 62: Compute the Euclidean distance between two arrays a and b

if exercise == 62:
    a = np.array([1,2,3,4,5])
    b = np.array([4,5,6,7,8])

    print(np.linalg.norm(a-b))

# Exercise 63: Find all the peaks in a 1D numpy array a. Peaks are points surrounded by smaller values on both sides

if exercise == 63:
    a = np.array([1, 3, 7, 1, 2, 6, 0, 1])

    doublediff = np.diff(np.sign(np.diff(a)))
    print(np.where(doublediff < 0)[0] + 1)

# Exercise 64: Subtract the 1d array b_1d from the 2d array a_2d, such that each item of b_1d subtracts from respective row of a_2d

if exercise == 64:
    a_2d = np.array([[3,3,3],[4,4,4],[5,5,5]])
    b_1d = np.array([1,1,1])

    print(a_2d - b_1d[:,None])

# Exercise 65: Find the index of the 5h repetition of number 1 in x

if exercise == 65:
    x = np.array([1, 2, 1, 1, 3, 4, 3, 1, 1, 2, 1, 1, 2])

    n = 5
    print(np.where(x==1)[0][n-1])

    # print([i for i, v in enumerate(x) if v ==1][n-1])

# Exercise 66: Convert numpy's datetime64 object to datetime's datetime object

if exercise == 66:
    dt64 = np.datetime64('2018-02-25 22:10:10')

    from datetime import datetime
    print(dt64.tolist())

    # print(dt64.astype(datetime))

# Exercise 67: Compute the moving average of window size 3 for the given 1D array

if exercise == 67:
    Z = np.random.randint(10, size=10)

    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n-1:]/n
    
    print(moving_average(Z, n=3).round(2))

    # print(np.convolve(Z, np.ones(3)/3, mode="valid").round(2))

# Exercise 68: Create a numpy array of length 10, starting from 5 and has a step of 3 between consecutive numbers

if exercise == 68:
    length = 10
    start = 5
    step = 3
    end = start + length * step
    out = np.arange(start, end, step)
    print(out)

# Exercise 69: Given an array of a non-continuous sequences of dates, make it a continuous sequence of dates by filling in the missing dates

if exercise == 69:
    dates = np.arange(np.datetime64('2018-02-01'), np.datetime64('2018-02-25'), 2)

    dates = np.arange(dates[0], dates[-1])
    print(dates)

    # filled_in = np.array(np.arange(data, (date+d) for date, d in zip(dates, np.diff(dates))]).reshape(-1))
    # output = np.hstack([filled_in, dates[-1]])
    # print(output)

    # out = []
    # for date, d in zip(dates, np.diff(dates)):
    #   out.append(np.arange(date, (date+d)))
    # filled_in = np.arrau(out).reshape(-1)
    # output = np.hstack([filled_in, dates[-1]])
    # print(output)

# Exercise 70: From the given 1D array, generate a 2D matrix using strides, with a window length of 4 and strides of 2

if exercise == 70:
    arr = np.arange(15)

    def gen_strides(a, stride_len=5, window_len=5):
        n_strides = ((a.size-window_len)//stride_len) + 1
        return np.array([a[s:(s+window_len)] for s in np.arange(0, n_strides*stride_len, stride_len)])
    
    print(gen_strides(arr, 2, 4))

# Exercises 71-101 were not published