import unittest
import os

import json
from jsonschema import (
    Draft201909Validator,
    Draft7Validator
)


from GTC import *

#-----------------------------------------------------
class TestArchiveJSONSchema135(unittest.TestCase):
    def test(self):

        schema_file = r"../GTC/schema/gtc_v_1_3_5.json"
        with open(schema_file,'r') as s:
            schema = json.load(s)

        _file = r"../test/ref_file_v_1_3_3.json"    
        with open(_file,'r') as f:
            file = json.load(f)

        self.assertTrue( 
            Draft201909Validator(schema).is_valid(file)
        )
        self.assertTrue( 
            Draft7Validator(schema).is_valid(file)
        )

#-----------------------------------------------------
class TestArchiveJSONFilev133(unittest.TestCase):

    """
    Use a reference file created using GTC v 1.3.3 
    to make sure we maintain backward compatibility.
    """
    
    def test(self):
        
        fname = 'ref_file_v_1_3_3.json'
        
        wdir =  os.path.dirname(__file__)
        path = os.path.join(wdir,fname)

        with open(path,'r') as f:
            ar = persistence.load_json(f)
            f.close()

        x1, y1, z1 = ar.extract('x','y','z')
        
        self.assertEqual( repr(x1), repr(ureal(1,1)) )
        self.assertEqual( repr(y1), repr(ureal(2,1)) )
        self.assertEqual( repr(z1), repr( ureal(1,1) + ureal(2,1) ) )

        z1 = ar.extract('z1')
        
        self.assertEqual( 
            repr(z1), 
            repr( 
                ureal(1,1,3) + ureal(2,1,4) 
            ) 
        )

        x1, y1, z1 = ar.extract('x2','y2','z2')

        x = ureal(1,1,independent=False)
        y = ureal(2,1,independent=False)
        r = 0.5
        set_correlation(r,x,y)

        self.assertEqual( repr(x1), repr(x) )
        self.assertEqual( repr(y1), repr(y) )
        self.assertEqual( repr(z1), repr(x+y) )

        x1, y1, z1 = ar.extract('x3','y3','z3')

        x,y = multiple_ureal([1,2],[1,1],4)     
        r = 0.5
        set_correlation(r,x,y)
        z = result( x + y )            
        self.assertEqual( repr(x1), repr(x) )
        self.assertEqual( repr(y1), repr(y) )
        self.assertEqual( repr(z1), repr(z) )
        
        self.assertEqual( get_correlation(x1,y1), r )

        x1, y1, z1 = ar.extract('x4','y4','z4')

        x = ucomplex(1,[10,2,2,10],5)
        y = ucomplex(1-6j,[10,2,2,10],7)
        
        z = result( log( x * y ) )
            
        self.assertEqual( repr(x1), repr(x) )
        self.assertEqual( repr(y1), repr(y) )
        self.assertEqual( repr(z1), repr(z) )

#-----------------------------------------------------
class TestArchivePickleFilev133(unittest.TestCase):

    """
    Use a reference file created using GTC v 1.3.3 
    to make sure we maintain backward compatibility.
    """
    
    def test(self):
        
        fname = 'ref_file_v_1_3_3.gar'
        
        wdir =  os.path.dirname(__file__)
        path = os.path.join(wdir,fname)

        with open(path,'rb') as f:
            ar = persistence.load(f)
            f.close()

        x1, y1, z1 = ar.extract('x','y','z')
        
        self.assertEqual( repr(x1), repr(ureal(1,1)) )
        self.assertEqual( repr(y1), repr(ureal(2,1)) )
        self.assertEqual( repr(z1), repr( ureal(1,1) + ureal(2,1) ) )

        z1 = ar.extract('z1')
        
        self.assertEqual( 
            repr(z1), 
            repr( 
                ureal(1,1,3) + ureal(2,1,4) 
            ) 
        )

        x1, y1, z1 = ar.extract('x2','y2','z2')

        x = ureal(1,1,independent=False)
        y = ureal(2,1,independent=False)
        r = 0.5
        set_correlation(r,x,y)

        self.assertEqual( repr(x1), repr(x) )
        self.assertEqual( repr(y1), repr(y) )
        self.assertEqual( repr(z1), repr(x+y) )

        x1, y1, z1 = ar.extract('x3','y3','z3')

        x,y = multiple_ureal([1,2],[1,1],4)     
        r = 0.5
        set_correlation(r,x,y)
        z = result( x + y )            
        self.assertEqual( repr(x1), repr(x) )
        self.assertEqual( repr(y1), repr(y) )
        self.assertEqual( repr(z1), repr(z) )
        
        self.assertEqual( get_correlation(x1,y1), r )

        x1, y1, z1 = ar.extract('x4','y4','z4')

        x = ucomplex(1,[10,2,2,10],5)
        y = ucomplex(1-6j,[10,2,2,10],7)
        
        z = result( log( x * y ) )
            
        self.assertEqual( repr(x1), repr(x) )
        self.assertEqual( repr(y1), repr(y) )
        self.assertEqual( repr(z1), repr(z) )
        
#============================================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'