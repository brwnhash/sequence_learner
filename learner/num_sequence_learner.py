
import os
import joblib
import numpy as np
from .common import ScalarToBitPattern
from core import TemproralLearner

class NumberSeqLearner():
    def __init__(self,start_num,end_num,store_path,train=False,config=None):
        self.seq_learner_path=os.path.join(store_path,'num_seq_learner.mm')
        self.sdr_converter_path=os.path.join(store_path,'sdr_converter.mm')        
        self.seq_start_symbol='ST'        
        bits_per_number=config['bits_per_number']
        num_neurons_per_col=config['neurons_per_col']
        overlap=config['word_overlap']
        num_active_cols=config['percent_active_col']
        self.offset_bits=bits_per_number
        self.numbers_map=self._get_numbers_map(start_num,end_num,bits_per_number,overlap)
        if train:           
            self.total_bits=self.sdr_converter.total_bits+self.offset_bits
            self.tm=TemproralLearner(self.numbers_map,self.total_bits,num_neurons_per_col,num_active_cols)
        else:
            self.tm=joblib.load(self.seq_learner_path)
            self.sdr_converter=joblib.load(self.sdr_converter_path)
            self.total_bits=self.sdr_converter.total_bits+self.offset_bits


    def _get_numbers_map(self,start_num,end_num,bits_per_number,overlap):

        num_divs=end_num-start_num
        self.sdr_converter=ScalarToBitPattern(start_num,end_num,num_divs,overlap,bits_per_number,True)
        num_range=list(range(start_num,end_num))
        total_bits=self.sdr_converter.total_bits+self.offset_bits  
        ptrns_map={}
        for idx,ptrn in zip(num_range,self.sdr_converter.all_ptrns):
            bit_ptrn=np.zeros(total_bits,dtype=int)
            col_idxs=np.nonzero(ptrn)
            col_idxs=np.add(col_idxs,self.offset_bits) #move location of bits
            np.put(bit_ptrn,col_idxs,1)
            ptrns_map[idx]=bit_ptrn
        bit_ptrn=np.zeros(total_bits,dtype=int)
        offset_elms=list(range(self.offset_bits))
        np.put(bit_ptrn,offset_elms,1)
        ptrns_map[self.seq_start_symbol]=bit_ptrn
        return ptrns_map

    def get_seq_ptrns(self,seqs):
        seq_ptrns_list=[]
        for seq in seqs:
            seq_ptrns=[]
            for idx,word in enumerate(seq):
                if idx==0:# add start symbol
                    bit_ptrn=self.numbers_map[self.seq_start_symbol]
                    seq_ptrns.append(bit_ptrn)
                bit_ptrn=self.numbers_map.get(word,[])
                seq_ptrns.append(bit_ptrn)
            seq_ptrns_list.append(seq_ptrns)
        return seq_ptrns_list

    def train(self,seqs,tags=None):
        seq_ptrns=self.get_seq_ptrns(seqs)          
        self.tm.build_sequences(seq_ptrns,seqs)
        joblib.dump(self.tm,self.seq_learner_path)
        joblib.dump(self.sdr_converter,self.sdr_converter_path)

    def forward_predict(self,seq,num_next_seq):
        seq_ptrns=self.get_seq_ptrns([seq])
        return self.tm.forward_seq_search(seq_ptrns[0],num_next_seq)





