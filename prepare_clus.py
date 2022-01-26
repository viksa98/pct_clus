TRAINSET = 'train_hecat.csv'      # path to the training set
TESTSET = 'test_hecat.csv'        # path to the test set
TIME_COLUMN = 'duration'                    # name of the column with time values
STATUS_COLUMN = 'truncated'                # name of the column with patient status
ID_COLUMN = 'id'                        # name of the ID column (not used for modeling)
NOMINAL_ATTRIBUTES = {'Gender': ['0', '1'],
'Profession_program': ['01',
'00',
'12',
'03',
'02',
'20',
'05',
'04',
'15',
'11',
'17',
'08',
'10',
'07',
'21',
'06',
'14',
'09',
'13',
'19',
'16',
'18',
'25',
'22',
'29',
'27',
'26',
'30',
'23',
'28',
'24'],
 'Education category': ['42',
  '25',
  '37',
  '48',
  '13',
  '43',
  '26',
  '30',
  '35',
  '21',
  '36',
  '17',
  '52',
  '31',
  '7',
  '51',
  '41',
  '9',
  '56',
  '47',
  '34'],
 'Dissabilities': ['0',
  '14',
  '10',
  '18',
  '12',
  '11',
  '6',
  '9',
  '7',
  '4',
  '13',
  '20',
  '16',
  '1'],
 'Reason for PES entry': ['8', '2', '16', '17', '12'],
 'eApplication': ['0', '1'],
 'Employment plan status': ['0', '4', '1', '5', '2'],
 'Employability assessment': ['0', '2', '1', '6'],
 'Employment plan ready': ['87',
  '36',
  '32',
  '34',
  '0',
  '40',
  '78',
  '39',
  '65',
  '37',
  '67',
  '52',
  '35',
  '38',
  '86',
  '84',
  '85',
  '64',
  '83',
  '22',
  '80',
  '57',
  '33',
  '58',
  '11',
  '3',
  '27',
  '68',
  '79',
  '81',
  '18',
  '21',
  '82',
  '1',
  '41',
  '62',
  '69',
  '15',
  '16',
  '23',
  '19']}


# find all unique timestamps in the training data
stamps = set()
with open(TRAINSET) as f:
    columns = f.readline().strip().split(',')
    time_idx = columns.index(TIME_COLUMN)

    for line in f:
        words = line.strip().split(',')
        if len(words) > 3:
            stamps.add(float(words[time_idx]) // 1)

stamps = sorted(list(stamps))


# Prepare data in arff format
for dset in [TRAINSET, TESTSET]:
    with open(dset[:-4]+'.arff', 'w') as f:
        print('@relation survival', file=f)

        with open(dset) as g:
            columns = g.readline().strip().split(',')
            n_cols = len(columns)
            time_idx = columns.index(TIME_COLUMN)
            status_idx = columns.index(STATUS_COLUMN)
            id_idx = columns.index(ID_COLUMN)

            # other columns
            for c in columns:
                if c in [TIME_COLUMN, STATUS_COLUMN]:
                    continue
                elif c == ID_COLUMN:
                    attr_type = 'key'
                elif c in NOMINAL_ATTRIBUTES:
                    attr_type = '{' + ', '.join(NOMINAL_ATTRIBUTES[c]) + '}'
                else:
                    attr_type = 'numeric'

                print(f'@attribute {c} {attr_type}', file=f)
                
            # status split by timestamps
            for s in stamps:
                print(f'@attribute time_{s} numeric', file=f)


            print('', file=f)
            print('@data', file=f)

            for line in g:
                words = line.strip().split(',')
                if len(words) > 3:
                    features = [words[id_idx]]
                    features += [words[i] for i in range(n_cols) if columns[i] not in [STATUS_COLUMN, TIME_COLUMN, ID_COLUMN]]
                    stamp = float(words[time_idx])
                    status = int(words[status_idx])
                    states = []
                    for t in stamps:
                        if t < stamp:
                            states.append('1')
                        elif status == 1:
                            states.append('0')
                        else:
                            states.append('?')
                    print(','.join(features+states), file=f)


# prepare the settings file for CLUS
n_features = n_cols - 2
n_stamps = len(stamps)
n_all = n_features + n_stamps

clustering_weights = [0] + [1]*(n_features-1) + [stamps[0]]
for i in range(1, len(stamps)):
    clustering_weights.append(stamps[i] - stamps[i-1])


settings_content = f"""
[General]
Verbose = 0

[Data]
File = {TRAINSET}.arff
TestSet = {TESTSET}.arff

[Attributes]
Key = 1
Descriptive = 2-{n_features}
Target = {n_features+1}-{n_all}
Clustering = 2-{n_all}
ClusteringWeights = {clustering_weights}

[Tree]
Heuristic = VarianceReduction
MissingClusteringAttrHandling = EstimateFromParentNode
MissingTargetAttrHandling = ParentNode
PruningMethod = M5

[Ensemble]
EnsembleMethod = RForest
Iterations = 100
SelectRandomSubspaces = SQRT
WriteEnsemblePredictions = Yes
NumberOfThreads = 8

[Output]
TrainErrors = Yes
TestErrors = Yes

[SemiSupervised]
SemiSupervisedMethod = PCT
PercentageLabeled = 100
PruningWhenTuning = No
InternalFolds = 3
%WeightScoresFile = weights.txt
PossibleWeights = [0.25,0.5,0.75]
"""

with open('settings.s', 'w') as f:
    print(settings_content, file=f)
