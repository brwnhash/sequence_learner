import numpy as np
import math

class ScalarToBitPattern():
    def __init__(self,start,end,divisions,overlap,single_field_bits,keep_ptrns=False):
        """
        start to end total distance . division- num of parts
        overlap- overlap between current and last value 
        single_field_bits- number of bits to represent single scalar value.
        """
        self.start,self.end,self.divisions=start,end,divisions
        self.overlap,self.single_field_bits=overlap,single_field_bits
        self.total_bits=0
        all_ptrns=self._val_range_to_bitfields(start,end,divisions,overlap,single_field_bits)
        self.all_ptrns=all_ptrns if keep_ptrns else []

    def _val_range_to_bitfields(self,start,end,divisions,overlap,single_field_bits):
        """
        if 10 bits taken and overlap 90% 1st to 2nd partion will match 90% and subsequent 
        drop
        """
        diff=end-start
        self.part_size=math.floor(diff/divisions)
        overlap_bits=math.floor(single_field_bits*overlap)
        self.shift_bits=(single_field_bits-overlap_bits)
        self.total_bits=self.shift_bits*(divisions-1)+single_field_bits
        #offset bits are for filing info between parts like 1 to 10 in between vals
        self.offset_bits=self.part_size/self.shift_bits
        start_idx=0
        bit_ptrn_list=[]
        part_info=[]  #only for debug
        curr_part=0
        default_bitptrn=np.zeros(self.total_bits,dtype=int)
        for part in range(start,end,self.part_size):
            end_idx=start_idx+single_field_bits
            set_ids=np.arange(start_idx,end_idx,1)
            bit_ptrn=default_bitptrn.copy()
            np.put(bit_ptrn,set_ids,1)
            part_info.append((start_idx,end_idx))
            start_idx+=self.shift_bits
            bit_ptrn_list.append(bit_ptrn)            
            curr_part+=1
        return bit_ptrn_list

    def to_bits(self,val,type='list'):
        bit_ptrn=np.zeros(self.total_bits,dtype=int)
        if  self.start<=val<=self.end:
            div,mod=int(val/self.part_size),int(val%self.part_size)
            start_idx=(div*self.shift_bits)+int((mod/self.offset_bits))
            set_ids=np.arange(start_idx,start_idx+self.single_field_bits,1)
            np.put(bit_ptrn,set_ids,1)
            return bit_ptrn.tolist() if type=='list' else bit_ptrn
        return Exception('bit field value not in range ')