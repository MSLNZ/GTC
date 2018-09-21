import weakref

class WeakSymmetricMatrix(object):

    """
    Implements a sparse symmetric matrix that
    maps object pairs to real-number 
    correlation coefficients.
    """

    def __init__(self):
        """
        WeakSymmetricMatrix() 
        
        """
        self._mat = weakref.WeakKeyDictionary()

    def __len__(self):
        """
        Return the number of rows
        
        """
        return len(self._mat)
        
    def get(self,p,default=0.0):
        x,y = p
        try:
            row_x = self._mat[x]
            return row_x.get(y,default)
        except KeyError:
            return default
            
    def __getitem__(self,p):
        x,y = p
        row_x = self._mat[x]
        return row_x[y]

    def __setitem__(self,p,r):
        x,y = p
        try:
            row_x = self._mat[x]
        except KeyError:
            # When creating a new row, we MUST set the diagonal element to 1,
            # because this assumed in some looping algorithms for variance 
            # and covariance
            self._mat[x] = row_x = weakref.WeakKeyDictionary({ x:1.0, y:r })
        else:
            row_x[y] = r
            
        if x!=y:
            # Create the symmetric element 
            try:
                row_y = self._mat[y]
            except KeyError:
                # When creating a new row, we MUST set the diagonal element to 1,
                # because this assumed in some looping algorithms for variance 
                # and covariance
                self._mat[y] = row_y = weakref.WeakKeyDictionary({ y:1.0, x:r })
            else:
                row_y[x] = r
        
    def __delitem__(self,p):
        x,y = p
        try:
            del self._mat[x][y]
            if x!=y:
                del self._mat[y][x]
        except KeyError:
            print "'{}' is not defined".format(p)

    def remove(self,id):
        """
        Remove all entries associated with `id`
        
        """
        if id in self._mat: del self._mat[id]
        
        for r in self._mat.keys():
            row = self._mat[r]
            if id in row: del row[id]

    def clear(self):
        """
        Clear the whole matrix
        
        """
        self._mat = weakref.WeakKeyDictionary()

    def submatrix(self,ids):
        """
        Return the submatrix with elements in `ids` 
        
        """
        R = WeakSymmetricMatrix()
        for i,id_i in enumerate(ids):
            if id_i in self._mat:
                row_i = self._mat[id_i]
                for id_j in ids[i+1:]:
                    if id_j in row_i:
                        R[id_i,id_j] = row_i[id_j]
                    
        return R         
        