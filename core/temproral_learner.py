
import scipy.sparse as sp
import numpy as np
from itertools import product,chain
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEBUG=False
def get_seq_pair_from_seqs(seqs,num_cols):
    """
    prev_seq and curr_seq pair,we are selecting NxN matrix
    idea is to get iteratively uneven lengths of seqs
    """
    if not isinstance(seqs,list):
        raise Exception('sequence must be instance of list as seqs are of variable length')
    num_seqs=len(seqs)
    empty_arr=[0]*num_cols
    col_idx=1
    while(True):
        prev_arr,curr_arr,num_curr_seq=[],[],0
        for seq_id in range(num_seqs):
            curr_seq=seqs[seq_id]
            if len(curr_seq)>col_idx:
                num_curr_seq+=1
                prev_arr.append(curr_seq[col_idx-1])
                curr_arr.append(curr_seq[col_idx])
            else:
                prev_arr.append(empty_arr)
                curr_arr.append(empty_arr)
        if not num_curr_seq:
            break
        yield sp.csr_matrix(prev_arr),sp.csr_matrix(curr_arr)
        num_curr_seq=0
        col_idx+=1

class TemproralLearner():
    def __init__(self,word_map,num_cols,num_neuron_per_col,num_active_cols=0.5,num_input_neurons=None):
        """
        same col selection: for chain like " a a a" to be recognized same col section
        is required ,we choose common columns and we choose next column in that case ,
        we may not require that if we doing for the cases where we doing probablistic
        inference .
        """
        self.num_cols=num_cols
        self.num_neuron_per_col=num_neuron_per_col
        self.total_neurons=num_neuron_per_col*num_cols
        #col_mat=np.full(self.total_neurons)
        self.num_connection_rows=num_input_neurons if num_input_neurons else self.total_neurons
        self.column_connections=sp.lil_matrix((self.num_connection_rows,self.total_neurons),dtype='float')
        self.col_to_neuron_idx=np.array([self._col_to_neurons(col_idx,num_neuron_per_col) for col_idx in range(num_cols)])
        self.last_run_connection_vertical_sum=np.array([])
        self.syn_wt_increment=0.2
        self.syn_wt_decrement=0.1
        self.init_syn_wt=1000
        self.num_active_cols=num_active_cols
        self.seq_number=0
        self.shuffle=True
        self._update_words_info(word_map)
            

    def _col_to_neurons(self,col_idx,n):
        start_idx=col_idx*n
        neurons=[start_idx+idx for idx in range(n)]
        return neurons

    def _shuffle_similar_score_idxs(self,scores,sorted_score_idx):
        """
        shuffle helps to assign different columns whenever there are similar
        score ties
        """
 
        new_idx_list,prev_score=np.array([]),-1
        stored_idxs=[]
        for idx in range(sorted_score_idx.size):
            curr_score=scores[idx]
            if curr_score==prev_score or idx==0:
                stored_idxs.append(sorted_score_idx[idx])
            else:
                np.random.shuffle(stored_idxs)
                new_idx_list=np.append(new_idx_list,stored_idxs)
                stored_idxs=[sorted_score_idx[idx]]
            prev_score=curr_score
        if stored_idxs:
            np.random.shuffle(stored_idxs)
            new_idx_list=np.append(new_idx_list,stored_idxs)
        return new_idx_list.astype(int)
       
    def _choose_neurons_in_cols(self,sorted_score_idx,neurons_per_row,used_cols):
        """
        every row choose N neurons which belong to different column.used_cols stores
        cols which are already used.
        """
        chosen_idxs=[]
        for idx in sorted_score_idx:
            col_idx=int(idx/self.num_neuron_per_col)
            if col_idx in used_cols:
                continue
            chosen_idxs.append(idx)
            used_cols.append(col_idx)
            if len(chosen_idxs)>=neurons_per_row:
                break
        return chosen_idxs

    def choose_least_used_cols(self,score_arr,curr_cols,n):
        """
        reshape array into N*num_neuron per col ,sum it to get least used cols
        """
        curr_arr=score_arr.reshape((curr_cols.size,self.num_neuron_per_col))
        curr_sum=np.sum(curr_arr,axis=1)
        sorted_idxs=np.argsort(curr_sum)
        if self.shuffle:
            sorted_idxs=self._shuffle_similar_score_idxs(curr_sum,sorted_idxs)
        actual_cols=curr_cols[sorted_idxs]
        return actual_cols[0:n]

    def _check_existing_connections(self,prev_neurons_idx,curr_neurons_idx,curr_neurons_connection_score,out_conn_per_input,used_cols,chosen_neurons_list):
        """
        check if previous neurons connected to any of new cols neurons.
        pick the one with maximum score..
        """
        num_exit_connections=0
        
        for idx in range(prev_neurons_idx.size):
            connection_score=curr_neurons_connection_score[idx]
            if connection_score.size==0:
                continue
            connection_score_ravel=connection_score.toarray().ravel()
            sorted_idxs=np.argsort(connection_score_ravel)
            #sort maximum score to minimum
            sorted_idxs=sorted_idxs[::-1]
            connection_score=connection_score_ravel[sorted_idxs]            
            sorted_idxs=curr_neurons_idx[sorted_idxs]              
            #choose maximum weight neurons
            chosen_neurons=self._choose_neurons_in_cols(sorted_idxs,out_conn_per_input,used_cols)
            chosen_neurons_list.append((idx,chosen_neurons))
            num_exit_connections+=len(chosen_neurons)

        return num_exit_connections

    def _check_new_connections(self,prev_neurons_idx,curr_neurons_connection_score,connection_vertical_sum,chosen_cols,out_conn_per_input,chosen_neurons_list,used_cols):
        """
        build new connections.prefer the cols with 
        among chosen cols ,take neurons with least score.
        """

        curr_neurons_idx=self.col_to_neuron_idx[chosen_cols]
        curr_neurons_idx=curr_neurons_idx.ravel()        
        connection_vertical_sum=connection_vertical_sum[curr_neurons_idx]
        sorted_col_score_idxs=np.argsort(connection_vertical_sum.ravel())
        if self.shuffle:
            shuffled_col_idxs=self._shuffle_similar_score_idxs(connection_vertical_sum,sorted_col_score_idxs)
        else:
            shuffled_col_idxs=sorted_col_score_idxs
        curr_idxs=curr_neurons_idx[shuffled_col_idxs]
        for idx in range(prev_neurons_idx.size):
            connection_score=curr_neurons_connection_score[idx]
            if connection_score.size!=0:
                continue           
            chosen_neurons=self._choose_neurons_in_cols(curr_idxs,out_conn_per_input,used_cols)
            chosen_neurons_list.append((idx,chosen_neurons))

    def _get_curr_cols_connection_score(self,prev_neurons_idx,curr_cols):
        """
        """
        
        connection_vertical_sum=np.sum(self.column_connections.toarray(),axis=0)
        curr_neurons_idx=self.col_to_neuron_idx[curr_cols]
        curr_neurons_idx=curr_neurons_idx.ravel()
        out_conn_per_input=1
        #out_conn_per_input=int(np.ceil(curr_cols.size/prev_neurons_idx.size))
        #out_conn_per_input=int(np.ceil(num_active_cols*out_conn_per_input))
        curr_neurons_connection_score=self.column_connections[prev_neurons_idx][:,curr_neurons_idx]
        chosen_neurons_list,used_cols=[],[]
        #existing connections
        num_exit_connections=self._check_existing_connections(prev_neurons_idx,curr_neurons_idx,curr_neurons_connection_score,out_conn_per_input,used_cols,chosen_neurons_list)
        #new connections
        if self.num_active_cols<1:
            Ncols=int(np.ceil(self.num_active_cols*curr_cols.size))
            #left_cols=Ncols-num_exit_connections
            chosen_cols=self.choose_least_used_cols(connection_vertical_sum[curr_neurons_idx],curr_cols,Ncols)        
        else:
            chosen_cols=curr_cols

        self._check_new_connections(prev_neurons_idx,curr_neurons_connection_score,connection_vertical_sum,chosen_cols,out_conn_per_input,chosen_neurons_list,used_cols)
         
        chosen_neurons_idxs,chosen_neuron_pairs=[],[]
        for idx,chosen_neurons in chosen_neurons_list:
            prev_idx=prev_neurons_idx[idx]
            current_idx=chosen_neurons
            chosen_neurons_idxs.extend(current_idx)
            chosen_neurons=list(product([prev_idx],current_idx))
            chosen_neuron_pairs.extend(chosen_neurons)
        chosen_neurons_idxs=np.sort(np.array(chosen_neurons_idxs))   
        return chosen_neuron_pairs,chosen_neurons_idxs

    def _get_winner_neurons_in_other_cols(self,prev_neurons_idx,curr_cols):
        """
        previous neurons idx to all current neurons idx scores are checked.
        [x1 and x2 prev_neuron] and [ x3,x4,x5 curr ] x3=sum(x1_conn+x2_conn) 
        in connection matrix that is column sum 
        arg_max is used if its different column where connection made
        arg_min is used if its same column where connection is made

        """
        return self._get_curr_cols_connection_score(prev_neurons_idx,curr_cols)
      

    def _do_score_increment(self,prev_neuron_idxs,chosen_neurons_list):
        rows,cols=zip(*chosen_neurons_list)
        rows,cols=list(rows),list(cols)
        curr_vals=np.asarray(self.column_connections[rows,cols].todense()).ravel()
        locs=np.where(curr_vals==0)[0]
        if locs.size!=0:
            curr_vals[locs]=self.init_syn_wt
        curr_vals=curr_vals+self.syn_wt_increment
        self.column_connections[rows,cols]=curr_vals

    def _do_score_decrement(self,row_connections,prev_neuron_idxs):
        curr_vals=np.asarray(row_connections.todense()).ravel()
        locs=np.where(curr_vals!=0)[0]
        if locs.size:
            vals=curr_vals[locs]-self.syn_wt_decrement
            rows,cols,div=[],[],row_connections.shape[1]
            for loc in locs:
                r_idx=prev_neuron_idxs[int(loc/div)]
                rows.append(r_idx)
                cols.append(loc%div)
            vals=np.clip(vals,1,1e5)
            self.column_connections[rows,cols]=vals

    def _update_connection_scores(self,prev_neuron_idxs,chosen_neurons_list):
        """
        scipy doesnt support inplace addition else np.add.at() can be used if dense matrix
        increment socre for new connections and reduce for old existing connections.
        down loops can be optimized.
        
        """
        self._do_score_increment(prev_neuron_idxs,chosen_neurons_list)
        row_connections=self.column_connections[prev_neuron_idxs]
        self._do_score_decrement(row_connections,prev_neuron_idxs)


    def _build_update_connections(self,prev_neurons_idx,curr_cols):
        chosen_neurons_list=[]
        if curr_cols.size:
            chosen_neurons_list,winner_neurons_idx=self._get_winner_neurons_in_other_cols(prev_neurons_idx,curr_cols)
        self._update_connection_scores(prev_neurons_idx,chosen_neurons_list)
        return winner_neurons_idx

    def _vec_to_row_col(self,curr_vecs):
        curr_cols,curr_rows=[],[]
        for idx,vec in enumerate(curr_vecs):
            indices=self.col_to_neuron_idx[vec.indices]
            indices=list(chain(*indices))
            curr_cols.extend(indices)
            curr_rows.extend([idx]*len(indices))
        return curr_rows,curr_cols

    def _vecs_to_csr_matrix(self,curr_vecs,offset):
        """
        spatial cols converted to neurons.offset is neuorn selected in column
        """
        actual_idx=[(ind*self.num_neuron_per_col)+offset[idx] for idx,ind in enumerate(curr_vecs.indices)]
        data,indices,ind_ptr=curr_vecs.data,actual_idx,curr_vecs.indptr
        curr_csr=sp.csr_matrix((data,indices,ind_ptr), shape=(curr_vecs.shape[0],self.total_neurons))
        return curr_csr

    def _get_curr_idx(self,curr_vecs,p_neuron_conn_score):
        """
        find maximum connection score to a column
        """
        curr_rows,curr_cols=self._vec_to_row_col(curr_vecs)
        curr_scores=p_neuron_conn_score[curr_rows,curr_cols]
        total_cols=curr_vecs.shape[0]*self.num_cols
        curr_scores=curr_scores.reshape((curr_vecs.nnz,self.num_neuron_per_col))        
        out_idxs=np.argmax(curr_scores,axis=1)
        out_idxs=np.asarray(out_idxs).ravel()
        curr_csr=self._vecs_to_csr_matrix(curr_vecs,out_idxs)
        return curr_csr

    def get_next_neurons_idx(self,prev_neuron_csr,curr_vecs):
        """
        array of prev_neurons_idx for N sequences
        curr_vecs:advanced indexing cant be used to fetch all info,as indices may be unequal
        some sparse array bigger than other
        """        
        p_neuron_conn_score=np.dot(prev_neuron_csr,self.column_connections)
        curr_neuron_csr=self._get_curr_idx(curr_vecs,p_neuron_conn_score)
        return curr_neuron_csr



    def _get_first_vec_idxs(self,prev_vec):
        idx_list=[]
        for vec in prev_vec:
            idx=vec.indices*self.num_neuron_per_col
            idx_list.append(idx)
        return np.array(idx_list)

    def _get_vec_to_csr(self,vec,val=np.array([])):
        c_idx=vec.indices
        indices=self.col_to_neuron_idx[c_idx].ravel()
        data=np.ones(indices.size) if val.size==0 else val.ravel()
        ind_ptr=np.array([0,indices.size])
        csr_matrix=sp.csr_matrix((data,indices,ind_ptr), shape=(1,self.total_neurons))
        return csr_matrix

    def _get_curr_prob_idxs(self,curr_vec,p_neuron_conn_score):
        """
        find maximum connection score to a column
        """
        curr_rows,curr_cols=self._vec_to_row_col(curr_vec)
        curr_scores=p_neuron_conn_score[curr_rows,curr_cols]      
        curr_csr=self._get_vec_to_csr(curr_vec,np.asarray(curr_scores))
        return curr_csr

    def get_next_prob_neurons_idxs(self,prev_neuron_csr,curr_vec=np.array([])):
        """
        all the connected neurons in sequence are found we dont select any particular
        neurons in columns ,its more like exploring ,its likely if next element of sequnece
        comes ,lot of old connections will be removed or weight reduced .
        return : next connected neurons matrix
        curr_vec :next columns are known ,neurons in column will be selected .
        if curr_vec is null then all the possible connections are returned.
        """        
        p_neuron_conn_score=np.dot(prev_neuron_csr,self.column_connections)
        p_neuron_conn_score.data=np.log(p_neuron_conn_score.data)
        if curr_vec.size==0:
            return p_neuron_conn_score
        curr_neuron_csr=self._get_curr_prob_idxs(curr_vec,p_neuron_conn_score)
        return curr_neuron_csr

    def get_prob_seq_neurons(self,seq):
        """
        first neuron we start with bursting .all the neurons in columns will be
        selected and we proceed from there
        """
        prev_csr_matrix=np.array([])
        for prev_vec,curr_vec in get_seq_pair_from_seqs([seq],self.num_cols):
            if not prev_csr_matrix.size:
                prev_csr_matrix=self._get_vec_to_csr(prev_vec)
            curr_csr_matrix=self.get_next_prob_neurons_idxs(prev_csr_matrix,curr_vec)
            prev_csr_matrix=curr_csr_matrix
        return prev_csr_matrix

    def forward_seq_search(self,seq,next_days,th=0.001,max_matches=100):
        """
        look for possible connection to current by backtracking.
        enable all bits of last column and look for possible assosiation in prev
        cols and choosing N cols in every step ,find the best connection.
        """
        prev_csr_matrix=self.get_prob_seq_neurons(seq)
        for c_day in range(0,next_days):
            prev_csr_matrix=self.get_next_prob_neurons_idxs(prev_csr_matrix)
        n_col=int(prev_csr_matrix.shape[1]/self.num_neuron_per_col)
        curr_arr=prev_csr_matrix.toarray()
        curr_arr=curr_arr.reshape((n_col,self.num_neuron_per_col))
        col_scores=np.max(curr_arr,axis=1)
        col_scores=col_scores.reshape((col_scores.size,1))
        sim_scores=np.dot(self.word_bit_ptrns,col_scores).ravel()
        #prob_scores=np.exp(sim_scores)/np.sum(np.exp(sim_scores))
        prob_scores=sim_scores/np.sum(sim_scores)
        sorted_idxs=np.argsort(prob_scores)
        sorted_idxs=sorted_idxs[::-1]
        sorted_idxs=sorted_idxs[0:max_matches]
        match_list=[]
        for idx in sorted_idxs:
            score=round(prob_scores[idx],6)
            if score<th:
                continue
            match_list.append({'match':self.words_map[idx],'score':score})
        return match_list

    def get_sequenence_active_neurons(self,seqs,callback):
        """
        seqs are N bit array
        """
        prev_csr_matrix=np.array([])
        for prev_vec,curr_vec in get_seq_pair_from_seqs(seqs,self.num_cols):
            if not prev_csr_matrix.size:
                offset=np.zeros(prev_vec.indices.shape,dtype=int)
                prev_csr_matrix=self._vecs_to_csr_matrix(prev_vec,offset)
                if callback:
                    callback(prev_csr_matrix)
            curr_csr_matrix=self.get_next_neurons_idx(prev_csr_matrix,curr_vec)
            if callback:
                callback(curr_csr_matrix)
            prev_csr_matrix=curr_csr_matrix


    
    def build_seq_memory(self,seq):
        """
        input seq : seq of input(csr_matrix)
        common_col:if connection is made to same col,we dont want to keep on increasing
        score,we do assosiation for new cols only.only one col will be chosen if same seq repeats
        """
     
        prev_vec=seq[0]
        if not isinstance(prev_vec,sp.csr_matrix):
            prev_vec=sp.csr_matrix(prev_vec)  
        Ncols=int(np.ceil(self.num_active_cols*prev_vec.indices.size))
        prev_neurons_idx=prev_vec.indices[0:Ncols]*self.num_neuron_per_col     
        seq_res=[] 
        for curr_vec in seq[1:]:
            if not isinstance(curr_vec,sp.csr_matrix):
                curr_vec=sp.csr_matrix(curr_vec) 
            prev_cols=prev_vec.indices
            curr_cols=curr_vec.indices
            curr_neurons_idx=self._build_update_connections(prev_neurons_idx,curr_cols)
            prev_vec,prev_neurons_idx=curr_vec,curr_neurons_idx
            seq_res.append(curr_neurons_idx)
            
        return seq_res

    def get_bit_patterns(self,seqs,callback):
        self.get_sequenence_active_neurons(seqs,callback)

    def _update_words_info(self,words_map):
        self.words_map,bit_ptrn_list={},[]
        for idx,(word,bit_ptrn) in enumerate(words_map.items()):
           self.words_map[idx]=word
           arr=[int(i) for i in bit_ptrn]
           bit_ptrn_list.append(arr)
        self.word_bit_ptrns=np.array(bit_ptrn_list)

    def _write_tmp_data(self,org_seq,res_list):
        with open('data/tmp_res.txt','w') as f:
            idx=0
            for seq,res in zip(org_seq,res_list):
                f.write('idx is '+str(idx)+'\n')
                f.write(str(seq)+'\n')
                f.write(str(res)+'\n')
                f.write('...\n')
                idx+=1

    def build_sequences(self,seqs,org_seq):
        res_list=[]
        for idx,seq in enumerate(seqs):  
            res=self.build_seq_memory(seq)
            res_list.append(res)
        if DEBUG:
            self._write_tmp_data(org_seq,res_list)        
        #self.get_bit_patterns(seqs,callback)
        return res_list






