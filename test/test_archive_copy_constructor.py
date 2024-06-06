import unittest
import os
import sys

import json

from lxml import etree

from GTC import *

#-----------------------------------------------------
class TestArchiveCopy(unittest.TestCase):

    def test(self):
        # ------------------------------
        # A do-nothing deepcopy
        ar = pr.Archive.copy( pr.Archive() )
        x1 = ureal(1,1,3,label='x1')
        x2 = ureal(5,2,6,label='x2')
        x3 = result(x1*x2)
        x4 = result(x1 + x3) 
        
        ar.add(x1=x1,x2=x2,x3=x3)
        
        s_rep = pr.dumps_json(ar)   
        
        # Frozen now
        self.assertRaises(RuntimeError,ar.add,x4=x4)    
        
        #------------------------------
        # For reading only 
        ar2 = pr.loads_json(s_rep)                        
        self.assertEqual( repr(x3), repr(ar2['x3']) )   

        self.assertRaises(RuntimeError,ar2.add,x4=x4)   
        
        #------------------------------
        # Should be able to add to a copy
        ar3 = pr.Archive.copy(ar)                         
        ar3.add(x4=x4)
        
        #------------------------------
        # No side effects for the copied archive!
        self.assertEqual(len(ar._tagged_real),3) 
        self.assertEqual(4,len(ar3._tagged_real)) 
        
        ar4 = pr.Archive.copy(ar2)  
        ar4.add(x4=x4)
        self.assertEqual(4,len(ar4._tagged_real)) 

    def test_1_5_0_json(self):
        fname = r'ref_file_v_1_5_0.json'
        wdir =  os.path.dirname(__file__)
        path = os.path.join(wdir,fname)

        with open(path,'r') as f:
            ar = persistence.load_json(f)
        
        ar2 = pr.Archive.copy(ar) 
        s_rep = persistence.dumps_json(ar2)


#============================================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'