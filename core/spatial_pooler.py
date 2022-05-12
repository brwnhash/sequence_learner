import numpy as np

#TODO : spatial pooler doesnt consider relation in input bits ,like if seq of bits
#are of column     
class SpatialPooler():
    def __init__(self,num_cols,num_input_bits,max_active_cols=None):
        """
        num_cols:size of output array.we used notation as columns as its spatial downsize
        as these cols further wil have neurons in temproral Pooler
        num_input_bits:size of input array.
        max_active_cols:num of output bits or colum that will be set.

        """
        self.num_cols=num_cols
        self.num_input_bits=num_input_bits
        self.max_active_cols=max_active_cols if max_active_cols else 0.2*num_cols
        self.col_connections=np.zeros((num_cols,num_input_bits),dtype=np.float32)
        self.syn_perm_update_val=0.1
        self.syn_perm_threshold=100
        self.syn_perm_min_val=3
        self.trained_input_ptrns=np.zeros((1,num_input_bits),dtype='int')

    def _choose_new_cols(self,connection_score,pre_connected_idxs,n):
        """
        select the one with the least score .For new cols, a score also must be
        set ,which is like threshold score 
        """
        least_connected_idxs=np.argsort(connection_score)
        new_idx= least_connected_idxs[0:n]
        return np.sort(new_idx)

    def _get_num_new_cols_to_choose(self,input_pattern,out_cols):
        """
        prev_non_zero_cols : all previous input connections. we want to penalize all input connections if they 
        dont occur in current connections.
        input: 100,110, now patter in 111 . we want to penalize first and second col 11 in 111
        as they are shared ,that is idea behind prev_non_zero_cols
        check how much is maximum overlap with existing input.
        num_cols=max_active_col*max_overlap(0 to 1)
        Note:

        """
        extra_cols,curr_non_zero_cols,new_cols,prev_non_zero_cols=1,[],[],[]
        if self.trained_input_ptrns.shape[0]>1:
            match=np.bitwise_and(input_pattern,self.trained_input_ptrns)
            score=np.sum(match,axis=0)
            prev_non_zero_cols=np.where(score>=1)[0]
            prev_non_zero_cols=list(prev_non_zero_cols)
            curr_non_zero_cols=np.where(input_pattern>=1)[0]
            new_cols=np.setdiff1d(curr_non_zero_cols,prev_non_zero_cols)          
            extra_cols=self.max_active_cols*(new_cols.size/curr_non_zero_cols.size)  
            new_cols=list(new_cols)
        else:
            new_cols=np.nonzero(input_pattern)[0]
            curr_non_zero_cols=new_cols
            extra_cols=self.max_active_cols
        self.trained_input_ptrns=np.vstack((self.trained_input_ptrns,input_pattern))
        _,_=new_cols.sort(),prev_non_zero_cols.sort()
        return int(np.max([1,extra_cols])),new_cols,prev_non_zero_cols
        

    def _get_col_connections(self,input_pattern,out_cols):
        """
        check the columns which are connected ,and choose new cols if number of
        connected cols are less
        """
        num_new_cols,new_input_cols,prev_non_zero_input_cols=self._get_num_new_cols_to_choose(input_pattern,out_cols)
        new_output_cols,pre_connected_output_cols=np.array([],dtype=int),np.array([],dtype=int)
        if num_new_cols:
            curr_col_connections=self.col_connections[out_cols] if out_cols.size else self.col_connections
            connection_score=np.sum(curr_col_connections,axis=1)
            pre_connected_output_cols=np.where(connection_score>=1)[0]
            pre_connected_output_cols=out_cols[pre_connected_output_cols] if out_cols.size else pre_connected_output_cols
            new_output_cols=self._choose_new_cols(connection_score,pre_connected_output_cols,self.max_active_cols)
            new_output_cols=out_cols[new_output_cols] if out_cols.size else new_output_cols

        return pre_connected_output_cols,new_output_cols,new_input_cols,prev_non_zero_input_cols

    def _get_unconnected_input_connections(self,prev_non_zero_input_cols,new_output_cols):
        """
        we checking which connection are unconnected in previously selected
        index,columns which we consider are not modified. we have input to output link ,its possible that from input to output link
        is not formed,we need to find that and set that link value at threshold
        """
        if not prev_non_zero_input_cols:
            return []
        col_connections=self.col_connections[new_output_cols]
        col_connections=col_connections[:,prev_non_zero_input_cols]==0
        col_connections.astype(int)
        #down loop not good,comeup with something better
        res=[]
        for connection in col_connections:
            rr=[prev_non_zero_input_cols[idx] for idx,cc in enumerate(connection) if cc]
            res.append(rr)
        return res

    def _update_unconnected_idx_scores(self,unconnected_idxs):
        curr_idx=0
        for cols in unconnected_idxs:
            if not cols:
                curr_idx+=1
                continue
            synapses_bit_ptrn=np.zeros(self.num_input_bits)
            np.put(synapses_bit_ptrn,cols,self.syn_perm_threshold)
            out_col=new_output_cols[curr_idx]
            self.col_connections[out_col]+=synapses_bit_ptrn
            curr_idx+=1

    def _update_col_connections(self,input_bit_ptrn,prev_connected_output_cols,new_output_cols,new_input_cols,prev_input_nonzero_cols):
        """
        update scores of each col connection
        new_cols: input cols which are new .
        prev_input_nonzero_cols :input non zero cols
        input col connection to every col input connection is like update every
        existing value.
        for new cols ,at input nonzero put syn_threhold value.
        at overlapping bit location score should be less.

        """
        new_output_cols,prev_connected_output_cols=new_output_cols.astype('int'),prev_connected_output_cols.astype('int')
        #nz_input_idx=np.nonzero(input_bit_ptrn)[0].astype('int')
        all_connected_idx=np.concatenate((new_output_cols,prev_connected_output_cols))
        #add new bit pattern at new col values..
        if new_output_cols.size:
            synapses_bit_ptrn=np.zeros(self.num_input_bits)
            np.put(synapses_bit_ptrn,new_input_cols,self.syn_perm_threshold)
            self.col_connections[new_output_cols]+=synapses_bit_ptrn
            unconnected_idxs=self._get_unconnected_input_connections(prev_input_nonzero_cols,new_output_cols)
            if unconnected_idxs:
                self._update_unconnected_idx_scores(unconnected_idxs)

        #reduce weight for cols in all the preconnected axis.
        if prev_input_nonzero_cols:
            synapses_bit_ptrn=np.zeros(self.num_input_bits)
            np.put(synapses_bit_ptrn,prev_input_nonzero_cols,self.syn_perm_update_val)
            self.col_connections[all_connected_idx]-=synapses_bit_ptrn

    def get_bit_patterns(self,input_patterns,output_columns,type='bool'):
        """
        check overlap with every input ,maximum overlap wins
        bitwise_and with columns gives only valid columns for given input.
        """
        num_input=input_patterns.shape[0]
        res=np.dot(self.col_connections,input_patterns.T)
        valid_connections=np.multiply(res,output_columns.T) if output_columns.size else res
        sorted_cols=np.argsort(valid_connections,axis=0)
        #start_idx=self.max_active_cols+1
        #sel_cols=sorted_cols[:-start_idx:-1]
        start_idx=sorted_cols.shape[0]-self.max_active_cols
        sel_cols=sorted_cols[start_idx:,:]
        active_bit_ptrns=[]
        for idx in range(num_input):
            col_conn,sel=valid_connections[:,idx],sel_cols[:,idx]
            vals=col_conn[sel] if type!='bool' else 1
            bit_arr=np.zeros(self.num_cols)
            np.put(bit_arr,sel,vals)
            active_bit_ptrns.append(bit_arr)
        return active_bit_ptrns
 
    def train(self,input_bit_patterns,output_columns_li=[]):
        """
        we learn on single training example iteratively and update weights.
        output_columns:cols of output for every input
        """
        self.output_columns=output_columns_li
        for idx,input_pattern in enumerate(input_bit_patterns):
            out_cols=output_columns_li[idx] if output_columns_li else np.array([])
            pre_connected_out_cols,new_out_cols,new_input_cols,prev_best_match_cols=\
                                self._get_col_connections(input_pattern,out_cols)
            self._update_col_connections(input_pattern,pre_connected_out_cols,new_out_cols,new_input_cols,prev_best_match_cols)

    def from_string_input_to_arr(self,input_ptrns):
        input_arr=[]
        for input in input_ptrns:
           arr=[i for i in input]
           input_arr.append(np.array(arr,dtype=int))
        return np.array(input_arr)

def _run_spatial_pooler(input_patterns,num_cols,max_active,output_cols,type='bool'):
    from itertools import chain
    num_input=len(input_patterns[0])
    sp=SpatialPooler(num_cols,num_input,max_active)
    input_patterns=sp.from_string_input_to_arr(input_patterns)
    res,col_ptrn=[],[]
    for col in output_cols:
        rr=[]
        for idx,c in enumerate(col):
            if int(c):
                rr.append(idx)
        bit_ptrn=np.zeros(num_cols,dtype=int)
        np.put(bit_ptrn,rr,1)
        col_ptrn.append(bit_ptrn)
        res.append(np.array(rr))

    sp.train(input_patterns,output_cols)
    output_col_ptrn=np.array(col_ptrn)
    sp.col_connections=np.ones((sp.num_cols,sp.num_input_bits),dtype=np.float32)
    rr=sp.get_bit_patterns(input_patterns,output_col_ptrn,type)
    return rr

def test_sp_isolated():
    input_patterns=['110000','001100','000011']  
    output_cols=['1010','1010','1100']
    rr=_run_spatial_pooler(input_patterns,4,3,output_cols,'float')
    print(rr)

def test_sp_overlapped():
    input_patterns=['1100','0110','0011']   
    rr=_run_spatial_pooler(input_patterns,2,2,[],'float')
    print(rr)

#test_sp_overlapped()
#test_sp_overlapped()

