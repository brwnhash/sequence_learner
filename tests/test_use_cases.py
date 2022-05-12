import os
from learner import NumberSeqLearner

def generate_tables(tbls,max_num):
    max_num=10
    mult_tbls={}
    for tbl in tbls:
        values=[(tbl*i) for i in range(1,max_num)]
        mult_tbls[tbl]=values
    return mult_tbls



def test_num_approx():
    seq_list=[]
    for i in range(0,5):
        seq_list.append([0,2,i,i+4,i+6,i+8])
    for i in range(0,5):
        seq_list.append([1,2,10+i,10+i+3])

    ns=NumberSeqLearner(0,50,'data/num_model',True)
    ns.train(seq_list)
    for i in range(0,2):
        val=seq_list[i][0:4]
        res=ns.forward_predict(val,1)
        print(res)

def test_num_avg():
    seq_list=[]
    for i in range(0,2):
        seq_list.append([3,0])
    seq_list.append([3,2])
    seq_list.append([3,4])
    print('starting')
    ns=NumberSeqLearner(0,10,'data/num_model',True)
    ns.train(seq_list)
    for i in range(0,2):
        val=seq_list[i][0]
        res=ns.forward_predict([val],1)
        print(res)

def train_predict_multipication_tables():
    model_dir='data'
    if not os.path.exists('data'):
        os.mkdir(model_dir)
    table_end=30
    max_cnt_per_table=10
    num_list=list(range(1,table_end+1))
    tables=generate_tables(num_list,max_cnt_per_table)
    print('starting ')
    config={'neurons_per_col':4,
        'word_overlap':0.2,
       'percent_active_col':0.7,
       'bits_per_number':5
       
       }
    min_val=0
    max_val=tables[table_end][-1]+1 #last value of multipication table we trying to learn
    ns=NumberSeqLearner(min_val,max_val,model_dir,True,config)
    for key,values in tables.items():
        print('training sequence ',values)
        ns.train([values])
    init_cnt=3
    next_values_to_pred=(max_cnt_per_table-init_cnt)
    num_errors=0
    for key,values in tables.items():
        values=values[0:init_cnt]
        print('......')
        print('next table for {tbl} and initial values given are {vv}'.format(tbl=key,vv=values))
        for i in range(1,next_values_to_pred):
            res=ns.forward_predict(values,i)
            print(res)
            actual_val=tables[key][init_cnt+i-1]
            if res[0]['match']!=actual_val:
                num_errors+=1
    print('number of prediction errors are ',num_errors)


