import numpy as np
from statistics import mean 

class NDCG:
    '''
    NDCG( length=20 )

    Used to iteratively calculate the Normalized Discounted Cumulative Gain for a result list with the defined length.

    Parameters
    -----------
    length : int
        NDCG@length
    '''

    def __init__(self, length=20):
        self.length = length
        self.test = 0
        self.pos = 0
    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.test = 0
        self.pos = 0


    def add_multiple(self, result, next_items, for_item=0, session=0,position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.

        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        dcg = self.dcg(result[:self.length].index, next_items)
        dcg_max = self.dcg(next_items[:self.length], next_items)

        self.pos += dcg/dcg_max
        self.test += 1


    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.

        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        self.add_multiple(result, [next_item])


    def dcg(self, result, next_items):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.

        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''

        # relatedItems = list(set(result) & set(next_items))
        # for i in range(len(relatedItems)):
        #     idx = list(result).index(relatedItems[i])+1 #ranked position = index+1
        #     if idx == 1:
        #         res += rel
        #     else:
        #         res += rel / np.log2(idx)

        res = 0;
        rel = 1;
        ranked_list_len = min(len(result), self.length)

        next_items = set(next_items)
        for i in range(ranked_list_len):          #range(self.length):
            if result[i] in next_items:
                if i == 0:
                    res += rel
                else:
                    res += rel / np.log2(i+1)

        # res = rel[0]+np.sum(rel[1:] / np.log2(np.arange(2, rel.size + 1)))
        return res



    def add_batch(self, result, next_item):
        '''
        Update the metric with a result set and the correct next item.

        Parameters
        --------
        result: pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.
        next_item: Array of correct next items
        '''
        i = 0
        for part, series in result.iteritems():
            result.sort_values(part, ascending=False, inplace=True)
            self.add(series, next_item[i])
            i += 1

    def score(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return self.pos/self.test




class MRR: 
    
    def __init__(self, length=20):
        self.length = length;
        self.MRR_score = [];
    def add(self, recommendation_list, next_item):
        
        res = recommendation_list[:self.length]
        if next_item in res.index:
            rank = res.index.get_loc( next_item ) + 1
            self.MRR_score.append(1.0/rank)    
        else:
            self.MRR_score.append(0)
            
    def score(self):
        return mean(self.MRR_score)
    
    def reset(self):
        self.MRR_score = []


class HR:
    def __init__(self, length=20):
        self.length = length
        self.HR_score = []
        self.totat_sessionsIn_data = 0
        
    def add(self, recommendation_list, next_item):
        res = recommendation_list[:self.length]
        if next_item in res.index:
            self.HR_score.append(1.0)    
        else:
            self.HR_score.append(0)
    def score(self):
        return mean(self.HR_score)   

    def reset(self):
        self.HR_score = []                 



class Precision: 
    '''
    Precision( length=20 )

    Used to iteratively calculate the average hit rate for a result list with the defined length. 

    Parameters
    -----------
    length : int
        HitRate@length
    '''
    
    def __init__(self, length=20):
        self.length = length;
    
    def init(self, train):
        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        return
        
    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.test=0;
        self.hit=0
    
    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        self.test += self.length
        if type(next_item) == list:
            self.hit += len( set(next_item) & set(result[:self.length].index) )
        else:
            self.hit += len( set([next_item]) & set(result[:self.length].index) )
    
    def add_multiple(self, result, next_items, for_item=0, session=0,position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        self.test += 1
        if type(next_items) == list:
            self.hit += len( set(next_items) & set(result[:self.length].index) ) / self.length
        
        else:
            self.hit += len( set([next_items]) & set(result[:self.length].index) ) / self.length
    def add_batch(self, result, next_item):
        '''
        Update the metric with a result set and the correct next item.
        
        Parameters
        --------
        result: pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.
        next_item: Array of correct next items
        '''
        i=0
        for part, series in result.iteritems(): 
            result.sort_values( part, ascending=False, inplace=True )
            self.add( series, next_item[i] )
            i += 1
        
    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("Precision@" + str(self.length) + ": "), (self.hit/self.test)