import unittest
import os
import sys

import json

from lxml import etree

from GTC import *

#-----------------------------------------------------
class TestArchiveJSONSchema135(unittest.TestCase):
    def test(self):

        wdir =  os.path.dirname(__file__)
        
        schema_file = r"../GTC/schema/gtc_v_1_5_0.json"
        _file = os.path.join(wdir,schema_file)       
        with open(_file,'r') as s:
            schema = json.load(s)

        fname = 'ref_file_v_1_5_0.json'        
        _file = os.path.join(wdir,fname)
        with open(_file,'r') as f:
            file = json.load(f)

        if sys.version_info >= (3,7):
            from jsonschema import Draft202012Validator
            # JSON schema adopted a different syntax for
            # tuple-arrays in the 2020-12 draft. 
            # It replaced `items` with `prefixItems`
            self.assertTrue( 
                Draft202012Validator(schema).is_valid(file)
            )

#-----------------------------------------------------
class TestArchiveJSONFilev150(unittest.TestCase):

    """
    Use a reference file created using GTC v 1.5.0
    to make sure we maintain backward compatibility.
    """
    
    def test(self):
        
        fname = 'ref_file_v_1_5_0.json'
        
        wdir =  os.path.dirname(__file__)
        path = os.path.join(wdir,fname)

        with open(path,'r') as f:
            ar = persistence.load_json(f)
            f.close()

        w, x, y, z = ar.extract('w','x','y', 'z')
        
        _w = ureal(1,1)
        _x = ureal(2,1)
        self.assertEqual( repr(w), repr(_w) )
        self.assertEqual( repr(x), repr(_x) )
        self.assertEqual( repr(y), repr( _w + _x ) )
        self.assertEqual( repr(z), repr( _x*(_w +_x) ) )
        self.assertEqual( component(z,w), 2)
        self.assertEqual( component(z,x), 5)
        self.assertEqual( component(z,y), 2*uncertainty(_w + _x))

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
class TestArchivePickleFilev150(unittest.TestCase):

    """
    Use a reference file created using GTC v 1.5.0
    to make sure we maintain backward compatibility.
    """
    
    def test(self):
        
        fname = 'ref_file_v_1_5_0.gar'
        
        wdir =  os.path.dirname(__file__)
        path = os.path.join(wdir,fname)

        with open(path,'rb') as f:
            ar = persistence.load(f)
            f.close()

        w, x, y, z = ar.extract('w','x','y', 'z')
        
        _w = ureal(1,1)
        _x = ureal(2,1)
        self.assertEqual( repr(w), repr(_w) )
        self.assertEqual( repr(x), repr(_x) )
        self.assertEqual( repr(y), repr( _w + _x ) )
        self.assertEqual( repr(z), repr( _x*(_w +_x) ) )
        self.assertEqual( component(z,w), 2)
        self.assertEqual( component(z,x), 5)
        self.assertEqual( component(z,y), 2*uncertainty(_w + _x))

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


class TestArchiveXMLSchema150(unittest.TestCase):

    def test(self):
        wdir = os.path.dirname(__file__)

        schema_file = r'../GTC/schema/gtc_v_1_5_0.xsd'
        _file = os.path.join(wdir, schema_file)
        schema = etree.XMLSchema(file=_file)

        fname = 'ref_file_v_1_5_0.xml'
        _file = os.path.join(wdir, fname)
        with open(_file, mode='rb') as f:
            source = etree.fromstring(f.read())

        schema.assertValid(source)


class TestArchiveXMLFilev150(unittest.TestCase):
    """
    Use a reference file created using GTC v 1.5.0
    to make sure we maintain backward compatibility.
    """

    def test(self):
        fname = 'ref_file_v_1_5_0.xml'

        wdir = os.path.dirname(__file__)
        path = os.path.join(wdir, fname)

        with open(path, 'rb') as f:
            ar = persistence.load_xml(f)

        w, x, y, z = ar.extract('w', 'x', 'y', 'z')

        _w = ureal(1, 1)
        _x = ureal(2, 1)
        self.assertEqual(repr(w), repr(_w))
        self.assertEqual(repr(x), repr(_x))
        self.assertEqual(repr(y), repr(_w + _x))
        self.assertEqual(repr(z), repr(_x * (_w + _x)))
        self.assertEqual(component(z, w), 2)
        self.assertEqual(component(z, x), 5)
        self.assertEqual(component(z, y), 2 * uncertainty(_w + _x))

        z1 = ar.extract('z1')

        self.assertEqual(
            repr(z1),
            repr(
                ureal(1, 1, 3) + ureal(2, 1, 4)
            )
        )

        x1, y1, z1 = ar.extract('x2', 'y2', 'z2')

        x = ureal(1, 1, independent=False)
        y = ureal(2, 1, independent=False)
        r = 0.5
        set_correlation(r, x, y)

        self.assertEqual(repr(x1), repr(x))
        self.assertEqual(repr(y1), repr(y))
        self.assertEqual(repr(z1), repr(x + y))

        x1, y1, z1 = ar.extract('x3', 'y3', 'z3')

        x, y = multiple_ureal([1, 2], [1, 1], 4)
        r = 0.5
        set_correlation(r, x, y)
        z = result(x + y)
        self.assertEqual(repr(x1), repr(x))
        self.assertEqual(repr(y1), repr(y))
        self.assertEqual(repr(z1), repr(z))

        self.assertEqual(get_correlation(x1, y1), r)

        x1, y1, z1 = ar.extract('x4', 'y4', 'z4')

        x = ucomplex(1, [10, 2, 2, 10], 5)
        y = ucomplex(1 - 6j, [10, 2, 2, 10], 7)

        z = result(log(x * y))

        self.assertEqual(repr(x1), repr(x))
        self.assertEqual(repr(y1), repr(y))
        self.assertEqual(repr(z1), repr(z))


#============================================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'