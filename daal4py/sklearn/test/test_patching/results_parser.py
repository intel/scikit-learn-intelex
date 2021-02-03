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
    fin = open('raw_log', 'r')
    mas = []
    for i in fin:
        if not i.startswith('INFO') and len(mas) != 0:
            run_parse(mas)
            mas.clear()
            mas.append(i.strip())
        else:
            mas.append(i.strip())
    for i in result:
        if result[i] != 'IDP':
            print(i, '->', result[i])
