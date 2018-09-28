class SymmetricMatrix(object):

    """
    The SymmetricMatrix class implements sparse symmetric 
    correlation matrices containing floating point numbers.
    """

    def __init__(self):
        """
        SymmetricMatrix() # construct an empty matrix
        
        """
        self._mat = {}

    def __len__(self):
        """Return the matrix dimension
        """
        return len(self._mat)
        
    def __getitem__(self,p):
        x,y = p
        try:
            row_x = self._mat[x]
            return row_x.get(y,0.0)
        except KeyError:
            return 0.0

    def __setitem__(self,p,r):
        x,y = p
        try:
            row_x = self._mat[x]
        except KeyError:
            # NB: set the diagonal element to 1
            # because this speeds up some looping
            # algorithms for variance and covariance
            self._mat[x] = row_x = { x:1.0, y:r }
        else:
            row_x[y] = r
        if x!=y:
            try:
                row_y = self._mat[y]
            except KeyError:
                self._mat[y] = row_y = { y:1.0, x:r }
            else:
                row_y[x] = r
        
    def __delitem__(self,p):
        x,y = p
        try:
            del self._mat[x][y]
            if x!=y:
                del self._mat[y][x]
        except KeyError:
            raise RuntimeError(
                "'{!s}' is not defined".format(p)
            )

    def remove(self,id):
        """
        Remove all entries associated with 'id'
        
        """
        if id in self._mat: del self._mat[id]
        
        for r in self._mat.keys():
            row = self._mat[r]
            if id in row: del row[id]

    def clear(self):
        """
        Clear all entries
        
        """
        self._mat = {}

    def submatrix(self,uids):
        """
        Return the submatrix for elements in `uid` 
        
        """
        R = SymmetricMatrix()
        for i,uid_i in enumerate(uids):
            if uid_i in self._mat:
                row_i = self._mat[uid_i]
                for uid_j in uids[i+1:]:
                    if uid_j in row_i:
                        R[uid_i,uid_j] = row_i[uid_j]
                    
        return R         
        