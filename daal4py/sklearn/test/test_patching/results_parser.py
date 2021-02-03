def make_unique(mas):
    s = set()
    ans = []
    for i in mas:
        if i not in s:
            ans.append(i)
            s.add(i)
    return ans


def get_method(s):
    cnt = 0
    ind = -1
    for i, elem in enumerate(s):
        if elem == '.':
            cnt += 1
        if cnt == 3:
            ind = i
            break
    assert ind != -1, "Bad string"
    method = ''
    for i in range(ind + 1, len(s)):
        if s[i] == ':':
            break
        method += s[i]
    return method


def get_branch(s):
    if len(s) == 0:
        return 'NO INFO'
    for i in s:
        if 'uses original Scikit-learn solver,' in i:
            return 'was in IDP, but go in Scikit'
    for i in s:
        if 'uses Intel(R) oneAPI Data Analytics Library solver' in i:
            return 'IDP'
    return 'Scikit'


result = {}


def run_parse(mas):
    name, dtype = mas[0].split()
    temp = []
    for i in range(1, len(mas)):
        mas[i] = mas[i][6:]
        if not mas[i].startswith('sklearn'):
            ind = name + ' ' + dtype + ' ' + mas[i]
            result[ind] = get_branch(temp)
            temp.clear()
        if mas[i].startswith('sklearn'):
            temp.append(mas[i])


TO_SKIP = [
    # --------------- NO INFO ---------------
    'KMeans int8 transform',
    'KMeans int8 score',
    'KMeans int16 transform',
    'KMeans int16 score',
    'KMeans int32 transform',
    'KMeans int32 score',
    'KMeans int64 transform',
    'KMeans int64 score',
    'KMeans float16 transform',
    'KMeans float16 score',
    'KMeans float32 transform',
    'KMeans float32 score',
    'KMeans float64 transform',
    'KMeans float64 score',
    'KMeans uint8 transform',
    'KMeans uint8 score',
    'KMeans uint16 transform',
    'KMeans uint16 score',
    'KMeans uint32 transform',
    'KMeans uint32 score',
    'KMeans uint64 transform',
    'KMeans uint64 score',
    'PCA int8 score',
    'PCA int16 score',
    'PCA int32 score',
    'PCA int64 score',
    'PCA float16 score',
    'PCA float32 score',
    'PCA float64 score',
    'PCA uint8 score',
    'PCA uint16 score',
    'PCA uint32 score',
    'PCA uint64 score',
    'LogisticRegression int8 decision_function',
    'LogisticRegression int16 decision_function',
    'LogisticRegression int32 decision_function',
    'LogisticRegression int64 decision_function',
    'LogisticRegression float16 decision_function',
    'LogisticRegression float32 decision_function',
    'LogisticRegression float64 decision_function',
    'LogisticRegression uint8 decision_function',
    'LogisticRegression uint16 decision_function',
    'LogisticRegression uint32 decision_function',
    'LogisticRegression uint64 decision_function',
    'LogisticRegressionCV int8 decision_function',
    'LogisticRegressionCV int8 predict',
    'LogisticRegressionCV int8 predict_proba',
    'LogisticRegressionCV int8 predict_log_proba',
    'LogisticRegressionCV int8 score',
    'LogisticRegressionCV int16 decision_function',
    'LogisticRegressionCV int16 predict',
    'LogisticRegressionCV int16 predict_proba',
    'LogisticRegressionCV int16 predict_log_proba',
    'LogisticRegressionCV int16 score',
    'LogisticRegressionCV int32 decision_function',
    'LogisticRegressionCV int32 predict',
    'LogisticRegressionCV int32 predict_proba',
    'LogisticRegressionCV int32 predict_log_proba',
    'LogisticRegressionCV int32 score',
    'LogisticRegressionCV int64 decision_function',
    'LogisticRegressionCV int64 predict',
    'LogisticRegressionCV int64 predict_proba',
    'LogisticRegressionCV int64 predict_log_proba',
    'LogisticRegressionCV int64 score',
    'LogisticRegressionCV float16 decision_function',
    'LogisticRegressionCV float16 predict',
    'LogisticRegressionCV float16 predict_proba',
    'LogisticRegressionCV float16 predict_log_proba',
    'LogisticRegressionCV float16 score',
    'LogisticRegressionCV float32 decision_function',
    'LogisticRegressionCV float32 predict',
    'LogisticRegressionCV float32 predict_proba',
    'LogisticRegressionCV float32 predict_log_proba',
    'LogisticRegressionCV float32 score',
    'LogisticRegressionCV float64 decision_function',
    'LogisticRegressionCV float64 predict',
    'LogisticRegressionCV float64 predict_proba',
    'LogisticRegressionCV float64 predict_log_proba',
    'LogisticRegressionCV float64 score',
    'LogisticRegressionCV uint8 decision_function',
    'LogisticRegressionCV uint8 predict',
    'LogisticRegressionCV uint8 predict_proba',
    'LogisticRegressionCV uint8 predict_log_proba',
    'LogisticRegressionCV uint8 score',
    'LogisticRegressionCV uint16 decision_function',
    'LogisticRegressionCV uint16 predict',
    'LogisticRegressionCV uint16 predict_proba',
    'LogisticRegressionCV uint16 predict_log_proba',
    'LogisticRegressionCV uint16 score',
    'LogisticRegressionCV uint32 decision_function',
    'LogisticRegressionCV uint32 predict',
    'LogisticRegressionCV uint32 predict_proba',
    'LogisticRegressionCV uint32 predict_log_proba',
    'LogisticRegressionCV uint32 score',
    'LogisticRegressionCV uint64 decision_function',
    'LogisticRegressionCV uint64 predict',
    'LogisticRegressionCV uint64 predict_proba',
    'LogisticRegressionCV uint64 predict_log_proba',
    'LogisticRegressionCV uint64 score',
    # --------------- Scikit ---------------
    'Ridge float16 predict',
    'Ridge float16 score',
    'RandomForestClassifier int8 predict_proba',
    'RandomForestClassifier int8 predict_log_proba',
    'RandomForestClassifier int16 predict_proba',
    'RandomForestClassifier int16 predict_log_proba',
    'RandomForestClassifier int32 predict_proba',
    'RandomForestClassifier int32 predict_log_proba',
    'RandomForestClassifier int64 predict_proba',
    'RandomForestClassifier int64 predict_log_proba',
    'RandomForestClassifier float16 predict_proba',
    'RandomForestClassifier float16 predict_log_proba',
    'RandomForestClassifier float32 predict_proba',
    'RandomForestClassifier float32 predict_log_proba',
    'RandomForestClassifier float64 predict_proba',
    'RandomForestClassifier float64 predict_log_proba',
    'RandomForestClassifier uint8 predict_proba',
    'RandomForestClassifier uint8 predict_log_proba',
    'RandomForestClassifier uint16 predict_proba',
    'RandomForestClassifier uint16 predict_log_proba',
    'RandomForestClassifier uint32 predict_proba',
    'RandomForestClassifier uint32 predict_log_proba',
    'RandomForestClassifier uint64 predict_proba',
    'RandomForestClassifier uint64 predict_log_proba',
]

if __name__ == '__main__':
    fin = open('daal4py/sklearn/test/test_patching/raw_log', 'r')
    mas = []
    for i in fin:
        if not i.startswith('INFO') and len(mas) != 0:
            run_parse(mas)
            mas.clear()
            mas.append(i.strip())
        else:
            mas.append(i.strip())
    for i in result:
        print(i, '->', result[i])
        if 'IDP' not in result[i]:
            assert i in TO_SKIP, 'Test patching failed: ' + i
