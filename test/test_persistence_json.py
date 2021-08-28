import unittest
import os
try:
    from itertools import izip  # Python 2
except ImportError:
    izip = zip

from GTC import *
from GTC.context import Context
from GTC import context
from GTC import persistence
from GTC.vector import is_ordered

from testing_tools import *

TOL = 1E-13 

#-----------------------------------------------------
class TestArchiveJSON(unittest.TestCase):
                         
    def test_with_file(self):
        """
        Save to a file and then restore by reading
        """
        wdir = os.getcwd()
        fname = 'test_file.json'
        path = os.path.join(wdir,fname)

        context._context = Context()
        x = ureal(1,1)
        y = ureal(2,1)
        z = result( x + y )

        ar = persistence.Archive()

        ar.add(x=x,y=y,z=z)

        f = open(path,'w')
        persistence.dump_json(f,ar)
        f.close()

        context._context = Context()
        f = open(path,'r')
        ar = persistence.load_json(f)
        f.close()
        os.remove(path)

        x1, y1, z1 = ar.extract('x','y','z')

        self.assertEqual( repr(x1), repr(x) )
        self.assertEqual( repr(y1), repr(y) )
        self.assertEqual( repr(z1), repr(z) )

    def test_with_file2(self):
        """
        Save to a file and then restore several times
        to test the effectiveness of GTC's uid system.
        
        """
        wdir = os.getcwd()
        fname = 'test_file.json'
        path = os.path.join(wdir,fname)

        context._context = Context()
        
        x = ureal(1,1,3,label='x')
        y = ureal(2,1,4,label='y')
        z = result( x + y )

        ar = persistence.Archive()

        # Saving only `z` means that when the archive
        # is restored `x` and `y` are not recreated as UNs.
        # However, Leaf nodes are created. We need to make sure 
        # that only one Leaf node gets created.
        
        ar.add(z=z)

        with open(path,'w') as f:
            persistence.dump_json(f,ar)

        context._context = Context()       
        with open(path,'r') as f:
            ar1 = persistence.load_json(f)
            z1 = ar1.extract('z')

        self.assertEqual( repr(z1), repr(z) )

        with open(path,'r') as f:
            # The attempt to create the uncertain number again is allowed 
            # but should not create a new node object 
            ar2 = persistence.load_json(f)
            z2 = ar2.extract('z')
            self.assertTrue( z2.is_intermediate )
            self.assertTrue( z1._node is z2._node )

        os.remove(path)
 
    def test_with_file3(self):
        """
        Dependent elementary UNs
        """
        wdir = os.getcwd()
        fname = 'test_file.json'
        path = os.path.join(wdir,fname)

        context._context = Context()

        x = ureal(1,1,independent=False)
        y = ureal(2,1,independent=False)
        
        r = 0.5
        set_correlation(r,x,y)
        
        z = result( x + y )
        
        ar = persistence.Archive()

        ar.add(x=x,y=y,z=z)

        with open(path,'w') as f:
            persistence.dump_json(f,ar)

        context._context = Context()
        with open(path,'r') as f:
            ar = persistence.load_json(f)
            
        os.remove(path)

        x1, y1, z1 = ar.extract('x','y','z')

        self.assertEqual( repr(x1), repr(x) )
        self.assertEqual( repr(y1), repr(y) )
        self.assertEqual( repr(z1), repr(z) )
        
        self.assertEqual( get_correlation(x,y), r )

    def test_with_file4(self):
        """
        Correlations with finite DoF
        """
        wdir = os.getcwd()
        fname = 'test_file.json'
        path = os.path.join(wdir,fname)

        context._context = Context()

        x,y = multiple_ureal([1,2],[1,1],4)
        
        r = 0.5
        set_correlation(r,x,y)
        
        z = result( x + y )
        
        ar = persistence.Archive()

        ar.add(x=x,y=y,z=z)

        with open(path,'w') as f:
            persistence.dump_json(f,ar)

        context._context = Context()
        with open(path,'r') as f:
            ar = persistence.load_json(f)

        os.remove(path)

        x1, y1, z1 = ar.extract('x','y','z')

        self.assertEqual( repr(x1), repr(x) )
        self.assertEqual( repr(y1), repr(y) )
        self.assertEqual( repr(z1), repr(z) )
        
        self.assertEqual( get_correlation(x1,y1), r )
 
    def test_with_file5(self):
        """
        Complex
        """
        wdir = os.getcwd()
        fname = 'test_file.json'
        path = os.path.join(wdir,fname)

        context._context = Context()

        x = ucomplex(1,[10,2,2,10],5)
        y = ucomplex(1-6j,[10,2,2,10],7)
        
        z = result( log( x * y ) )
        
        ar = persistence.Archive()

        ar.add(x=x,y=y,z=z)

        with open(path,'w') as f:
            persistence.dump_json(f,ar)

        context._context = Context()
        with open(path,'r') as f:
            ar = persistence.load_json(f)

        os.remove(path)

        x1, y1, z1 = ar.extract('x','y','z')

        self.assertEqual( repr(x1), repr(x) )
        self.assertEqual( repr(y1), repr(y) )
        self.assertEqual( repr(z1), repr(z) )
        
    def test_with_string1(self):
        """
        Simple save with intermediate 
        """
        context._context = Context()

        x = ureal(1,1)
        y = ureal(2,1)
        z = result( x + y )
        
        ar = persistence.Archive()

        ar.add(x=x,y=y,z=z)

        db = persistence.dumps_json(ar)

        context._context = Context()
        ar = persistence.loads_json(db)

        x1, y1, z1 = ar.extract('x','y','z')

        self.assertEqual( repr(x1), repr(x) )
        self.assertEqual( repr(y1), repr(y) )
        self.assertEqual( repr(z1), repr(z) )


    def test_with_string2(self):
        """
        Save to a file and then restore several times
        to test the effectiveness of GTC's uid system.
        
        """
        context._context = Context()
        
        x = ureal(1,1,3,label='x')
        y = ureal(2,1,4,label='y')
        z = result( x + y )

        ar = persistence.Archive()

        # Saving only `z` means that when the archive
        # is restored `x` and `y` are not recreated as UNs.
        # However, Leaf nodes are created. We need to make sure 
        # that only one Leaf node gets created.
        
        ar.add(z=z)

        s = persistence.dumps_json(ar)

        context._context = Context() 
        
        ar1 = persistence.loads_json(s)
        z1 = ar1.extract('z')

        # The attempt to create a new uncertain number
        # is allowed but a new node is not created
        ar2 = persistence.loads_json(s)
        z2 = ar2.extract('z')
        self.assertTrue(z2.is_intermediate)
        self.assertTrue( z2._node is z1._node)
        
    def test_with_string3(self):
        """
        Dependent elementary UNs
        """
        context._context = Context()

        x = ureal(1,1,independent=False)
        y = ureal(2,1,independent=False)
        
        r = 0.5
        set_correlation(r,x,y)
        
        z = result( x + y )
        
        ar = persistence.Archive()

        ar.add(x=x,y=y,z=z)

        db = persistence.dumps_json(ar)

        context._context = Context()
        ar = persistence.loads_json(db)

        x1, y1, z1 = ar.extract('x','y','z')

        self.assertEqual( repr(x1), repr(x) )
        self.assertEqual( repr(y1), repr(y) )
        self.assertEqual( repr(z1), repr(z) )
         
        self.assertEqual( get_correlation(x,y), r )

    def test_with_string4(self):
        """
        Correlations with finite DoF
        """
        context._context = Context()

        x,y = multiple_ureal([1,2],[1,1],4)
        
        r = 0.5
        set_correlation(r,x,y)
        
        z = result( x + y )
        
        ar = persistence.Archive()

        ar.add(x=x,y=y,z=z)

        db = persistence.dumps_json(ar)

        context._context = Context()
        ar = persistence.loads_json(db)

        x1, y1, z1 = ar.extract('x','y','z')

        self.assertEqual( repr(x1), repr(x) )
        self.assertEqual( repr(y1), repr(y) )
        self.assertEqual( repr(z1), repr(z) )
        
        self.assertEqual( get_correlation(x,y), r )
 
    def test_with_string5(self):
        """
        Complex
        """
        context._context = Context()

        x = ucomplex(1,[10,2,2,10],5)
        y = ucomplex(1-6j,[10,2,2,10],7)
        
        z = result( log( x * y ) )
        
        ar = persistence.Archive()

        ar.add(x=x,y=y,z=z)

        db = persistence.dumps_json(ar)

        context._context = Context()
        ar = persistence.loads_json(db)

        x1, y1, z1 = ar.extract('x','y','z')

        self.assertEqual( repr(x1), repr(x) )
        self.assertEqual( repr(y1), repr(y) )
        self.assertEqual( repr(z1), repr(z) )
        

#============================================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'