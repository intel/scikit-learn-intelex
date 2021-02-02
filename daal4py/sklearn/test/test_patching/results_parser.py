def make_unique(mas):
    s = set()
    ans = []
    for i in mas:
        if i not in s:
            ans.append(i)
            s.add(i)
    return ans


def get_method(s):
    s = s[6:]
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
    if 'uses Intel(R) oneAPI Data Analytics Library solver' in s:
        return 'IDP'
    if 'uses original Scikit-learn solver,' in s:
        return 'IDP -> Scikit'
    return 'Scikit'


result = {}


def run_parse(mas):
    mas = make_unique(mas)
    name, dtype = mas[0].split()
    for i in range(1, len(mas)):
        ind = name + ' ' + dtype + ' ' + get_method(mas[i])
        if ind in result:
            continue
        result[ind] = 'IDP' in get_branch(mas[i])


TO_SKIP = [
    'RandomForestClassifier int8 predict_proba',
    'RandomForestClassifier int16 predict_proba',
    'RandomForestClassifier int32 predict_proba',
    'RandomForestClassifier int64 predict_proba',
    'RandomForestClassifier float16 predict_proba',
    'RandomForestClassifier float32 predict_proba',
    'RandomForestClassifier float64 predict_proba',
    'RandomForestClassifier uint8 predict_proba',
    'RandomForestClassifier uint16 predict_proba',
    'RandomForestClassifier uint32 predict_proba',
    'RandomForestClassifier uint64 predict_proba',
    'LinearRegression int8 predict',
    'LinearRegression int16 predict',
    'LinearRegression int32 predict',
    'LinearRegression int64 predict',
    'LinearRegression float16 predict',
    'Ridge float16 predict',
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
        branch = 'IDP' if result[i] is True else 'Scikit'
        print(i, '->', branch)
        if result[i] is False:
            assert i in TO_SKIP, 'Test patching failed: ' + i
