
import numpy as np

def OCRAccuracy(weight, examples, labels, d):
    """
        CALCULATES THE ACCURACY MEASUREMENTS OF OCR IN THE OUTPUT FILE.
    """
    count = 0
    mySet = np.shape(examples)
    for i in range(0, mySet[0]):
        p = np.zeros((1, alphabets_count))
        for j in range(0, alphabets_count):
            p[0][j] = np.dot(examples[i], np.transpose(weight[j]))
        if d[np.argmax(p)] == labels[i]:
            count += 1
    return count / mySet[0]

def OCR_outputResults(numOfIterations, outfile, OCR_a_test, OCR_a_train, mistakes, test, train):
    """
        PRINTS THE ACCURACY MEASUREMENTS OF OCR IN THE OUTPUT FILE.
    """
    f = open(outfile, 'a')
    f.write("\n\n***** OCR OUTPUT ***** \n\n")
    for i in range(1, numOfIterations + 1):
        f.write(str(i) + ' ' + str(mistakes[i - 1]) + '\n')
    for i in range(1, numOfIterations + 1):
        f.write(str(i) + ' ' + str(train[i - 1]) + ' ' + str(test[i - 1]) + '\n')
    f.write(str(train[numOfIterations - 1]) + ' ' + str(test[numOfIterations - 1]) + '\n')
    f.write(str(OCR_a_train) + ' ' + str(OCR_a_test) + '\n')
    f.close()

def OCR_averagePerceptron(alphabets_count, learningRate, mySet, features, index_to_letter, letter_to_index, testLabel, trainLabel):
    """
        AVERAGE PERCEPTRON FOR THE OCR
    """
    weights = np.zeros((alphabets_count, features + 1))
    used = np.zeros((alphabets_count, features + 1))
    count = 1
    for i in range(1, alphabets_count + 1):
        for j in range(0, mySet[0]):
            predicted = np.zeros((1, alphabets_count))
            for k in range(0, alphabets_count):
                predicted[0][k] = np.dot(train_data[j], np.transpose(weights[k]))
            p_i = np.argmax(predicted)
            a_i = letter_to_index[trainLabel[j]]
            if p_i != a_i:
                weights[p_i] = weights[p_i] - learningRate * train_data[j]
                weights[a_i] = weights[a_i] + learningRate * train_data[j]
                used[p_i] = used[p_i] - count * learningRate * train_data[j]
                used[a_i] = used[a_i] + count * learningRate * train_data[j]
            count += 1
    weights = weights - used * (1 / count)
    OCR_a_train = OCRAccuracy(weights, train_data, trainLabel, index_to_letter)
    OCR_a_test =  OCRAccuracy(weights, test_data, testLabel, index_to_letter)
    return OCR_a_test, OCR_a_train


def outputResult(averageTestAccuracy, averageTrainAccuracy, mistakes, test, train):
    """
        PRINTS THE ACCURACY MEASUREMENTS OF FORTUNE COOKIE IN THE OUTPUT FILE.
    """
    f = open(outputfile, 'w')
    f.write("***** FORTUNE COOKIE OUTPUT ***** \n\n")
    for i in range(0, numOfIterations):
        f.write(str(i) + ' ' + str(mistakes[i - 1]) + '\n')
    for i in range(1, numOfIterations + 1):
        f.write(str(i) + ' ' + str(train[i - 1]) + ' ' + str(test[i - 1]) + '\n')
    f.write(str(train[numOfIterations - 1]) + ' ' + str(test[numOfIterations - 1]) + '\n')
    f.write(str(averageTrainAccuracy) + ' ' + str(averageTestAccuracy) + '\n\n')
    f.close()

def accuracy(weight, cookies, fortune_training_labels):
    """
        CALCULATES THE ACCURACY MEASUREMENTS OF FORTUNE COOKIE IN THE OUTPUT FILE
    """
    c = 0
    mySet = np.shape(cookies)
    for index in range(0, mySet[0]):
        predictedLabel = np.dot(cookies[index], np.transpose(weight))
        if ((predictedLabel[0] > 0 and fortune_training_labels[index] > 0) or \
                (predictedLabel[0] <= 0 and fortune_training_labels[index] < 0)):
            c += 1
    return c / mySet[0]

def averagePerceptron(numberOfWords, cookies, cookieTestingData,cookieTestingLabel, cookietrainLabel):
    """
        AVERAGE PERCEPTRON FOR THE FORTUNE COOKIE
    """
    weights = np.zeros((1, numberOfWords + 1))
    u = np.zeros((1, numberOfWords + 1))
    c = 1
    S = np.shape(cookies)
    for i in range(1, numOfIterations + 1):
        for index in range(0, S[0]):
            predicted = np.dot(cookies[index], np.transpose(weights))
            if predicted[0] * cookietrainLabel[index] <= 0:
                weights = weights + learningRate * cookietrainLabel[index] * cookies[index]
                u = u + c * learningRate * cookietrainLabel[index] * cookies[index]
            c += 1
    weights = weights - u * (1 / c)
    averageTestAccuracy = accuracy(weights, cookies, cookietrainLabel)
    averageTrainAccuracy = accuracy(weights, cookieTestingData, cookieTestingLabel)
    return averageTestAccuracy, averageTrainAccuracy

def simplePerceptron(numberOfWords, cookies, fortune_testing_data,fortune_test_labels, fortune_training_labels):
    """
        SIMPLE PERCEPTRON FOR THE FORTUNE COOKIE
    """
    weights = np.zeros((1, numberOfWords + 1))  # weights
    max,train,test = [],[],[]
    mySet = np.shape(cookies)
    for i in range(1, numOfIterations + 1):
        mistakes = 0
        for index in range(0, mySet[0]):
            predicted = np.dot(cookies[index], np.transpose(weights))
            if predicted[0] * fortune_training_labels[index] <= 0:
                mistakes += 1
                weights = weights + learningRate * fortune_training_labels[index] * cookies[index]
        max.append(mistakes)
        train.append(accuracy(weights, cookies, fortune_training_labels))
        test.append(accuracy(weights, fortune_testing_data, fortune_test_labels))
    return max, test, train

def OCR_simplePerceptron(alphabets_count, numOfIterations, learningRate, outputfile, features, index_to_letter, letter_to_index, testLabel, trainLabel, test_data, train_data):
    """
        SIMPLE PERCEPTRON FOR THE OCR.
    """
    weights = np.zeros((alphabets_count, features + 1))
    mistakeLi = []
    train = []
    test = []
    mySet = np.shape(train_data)
    for i in range(1, numOfIterations + 1):
        mistakes = 0
        for j in range(0, mySet[0]):
            predicted = np.zeros((1, alphabets_count))
            for k in range(0, alphabets_count):
                predicted[0][k] = np.dot(train_data[j], np.transpose(weights[k]))
            p_i = np.argmax(predicted)
            a_i = letter_to_index[trainLabel[j]]
            if p_i != a_i:
                mistakes += 1
                weights[p_i] = weights[p_i] - learningRate * train_data[j]
                weights[a_i] = weights[a_i] + learningRate * train_data[j]
        mistakeLi.append(mistakes)
        train.append(OCRAccuracy(weights, train_data, trainLabel, index_to_letter))
        test.append(OCRAccuracy(weights, test_data, testLabel, index_to_letter))
        return train, test, mistakeLi

#################### INITIALIZING THE FILE VARIBALES ####################
cookieTrainData = 'traindata.txt'
cookietrainLabel = 'trainlabels.txt'
cookieTestingData = 'testdata.txt'
cookieTestingLabel = 'testlabels.txt'
stopWords = 'stoplist.txt'
outputfile = 'output.txt'
learningRate = 1
numOfIterations = 20
alphabets_count = 26
ocrTrain = 'ocr_train.txt'
ocrTest = 'ocr_test.txt'

# STORING UNIQUE WORDS FROM THE COOKIE TRAINING FILE
cookieTrainWords = set()
f = open(cookieTrainData, 'r')
lines = f.read().split('\n')
for line in lines:
    words = line.split(' ')
    for word in words:
        cookieTrainWords.add(word)
f.close()

# DELETING ALL THE STOP WORDS FROM THE TRAINING DATA.
f = open(stopWords, 'r')
stop_words = f.read().split('\n')
for word in stop_words:
    cookieTrainWords.discard(word)
f.close()

# SORTING THE WORDS IN THE LIST AND THEN STORING THEM IN THE DICTIONARY WITH VALUE AS THEIR INDEX.
vocabulary = dict()
numberOfWords = len(cookieTrainWords)
for index, word in enumerate(sorted(cookieTrainWords)):
    vocabulary[word] = index

# EXTRACTING THE FEATURES FROM THE TRAINING DATA.
cookies = np.zeros((len(lines), numberOfWords + 1))
for index, line in enumerate(lines):
    words = line.split(' ')
    for word in words:
        if vocabulary.get(word) is not None:
            cookies[index][vocabulary[word]] = 1
    cookies[index][numberOfWords] = 1
del lines

# READING THE TRANING LABELS AND FILTERING THE DATA AS +1 OR -1
f = open(cookietrainLabel, 'r')
fortune_training_labels = f.read().split('\n')
for index, label in enumerate(fortune_training_labels):
    if int(label) == 0:
        fortune_training_labels[index] = -1
    else:
        fortune_training_labels[index] = 1
f.close()

# CREATING THE TESTING VECTORS
f = open(cookieTestingData, 'r')
lines = f.read().split('\n')
lines.pop()
fortune_testing_data = np.zeros((len(lines), numberOfWords + 1))
for index, line in enumerate(lines):
    words = line.split(' ')
    for word in words:
        if vocabulary.get(word) is not None:
            fortune_testing_data[index][vocabulary[word]] = 1
    fortune_testing_data[index][numberOfWords] = 1

f.close()


# READING THE TESTING LABELS FROM THE FILE.
f = open(cookieTestingLabel, 'r')
fortune_test_labels = f.read().split('\n')
fortune_test_labels.pop()
for index, label in enumerate(fortune_test_labels):
    if int(label) == 0:
        fortune_test_labels[index] = -1
    else:
        fortune_test_labels[index] = 1
f.close()

mistakes, test, train = simplePerceptron(numberOfWords, cookies, fortune_testing_data,fortune_test_labels, fortune_training_labels)
averageTestAaccuracy, averageTrainAaccuracy = averagePerceptron(numberOfWords, cookies, fortune_testing_data,fortune_test_labels, fortune_training_labels)
outputResult(averageTestAaccuracy, averageTrainAaccuracy, mistakes, test, train)


# OCR QUESTION

trainData = []
trainLabel = []
f = open(ocrTrain, 'r')
lines = f.read().split('\n')
for line in lines:
    elements = line.split('\t')
    if (len(elements) > 3) and (elements[3] == '_'):
        trainData.append(elements[1].lstrip('im'))
        trainLabel.append(elements[2])
f.close()

# EXTRACTING TEST LABELS
testData = []
testLabel = []
f = open(ocrTest, 'r')
lines = f.read().split('\n')
for line in lines:
    elements = line.split('\t')
    if (len(elements) > 3) and (elements[3] == '_'):
        testData.append(elements[1].lstrip('im'))
        testLabel.append(elements[2])
f.close()
features = len(trainData[0])
train_data = np.zeros((len(trainData), features + 1))
for index, example in enumerate(trainData):
    for i, digit in enumerate(example):
        train_data[index][i] = int(digit)
    train_data[index][features] = 1



# EXTRACTING THE TEST DATA
test_data = np.zeros((len(testData), features + 1))
for index, example in enumerate(testData):
    for i, digit in enumerate(example):
        test_data[index][i] = int(digit)
    test_data[index][features] = 1
del testData
letter_to_index = dict()
index_to_letter = dict()
letters = sorted(list(set(trainLabel)))
for index, letter in enumerate(letters):
    letter_to_index[letter] = index
    index_to_letter[index] = letter

# OCR SIMPLE PERCEPTRON CALL.
OCR_s_train, OCR_s_test, OCR_s_mistakes = OCR_simplePerceptron(alphabets_count, numOfIterations, learningRate, outputfile, features, index_to_letter, letter_to_index, testLabel, trainLabel, test_data, train_data)
mySet = np.shape(trainData)

# OCR AVERAGE PERCEPTRON CALL
OCR_a_test, OCR_a_train = OCR_averagePerceptron(alphabets_count, learningRate, mySet, features, index_to_letter, letter_to_index, testLabel, trainLabel)
OCR_outputResults(numOfIterations, outputfile, OCR_a_test, OCR_a_train, mistakes, test, train)
