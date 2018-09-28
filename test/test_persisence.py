import unittest
import sys
import math
import cmath
import itertools

from GTC import *
from GTC.context import Context

from testing_tools import *

TOL = 1E-13 

#-----------------------------------------------------
class TestArchive(unittest.TestCase):

    def test_context_management(self):
        """
        Context registers should be updated when all dependence
        on an influence quantity disappears.
        
        """
        x1 = ureal(1,1,5,label='x1',independent=False)
        x2 = ureal(1,1,5,label='x2',independent=False)
        x3 = ureal(1,1,5,label='x3',independent=False)

        set_correlation(.5,x1,x2)
        set_correlation(.5,x2,x3)
        
        x4 = x1 + x2
        x5 = x2 + x3
        x6 = x4 + x5

        ar = archive.Archive()
        ar.add(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5,x6=x6)
        ar._freeze()

        # unpack and test
        context = Context()
        ar._thaw(context)
        y1, y2, y3 = ar.extract('x1','x2','x3')
        y4, y5, y6 = ar.extract('x4','x5','x6')

        self.assertEqual(x1.s,y1.s)
        self.assertEqual(x2.s,y2.s)
        self.assertEqual(x3.s,y3.s)
        self.assertEqual(x4.s,y4.s)
        self.assertEqual(x5.s,y5.s)
        self.assertEqual(x6.s,y6.s)

        uid1 = y1._uid
        uid2 = y2._uid
        uid3 = y3._uid

        # self.assert_ ( uid1 in context._dof_record )
        # self.assert_ ( uid2 in context._dof_record )
        # self.assert_ ( uid3 in context._dof_record )
        
        # self.assert_ ( y1._node in context.labels )
        # self.assert_ ( y2._node in context.labels )
        # self.assert_ ( y3._node in context.labels )

        self.assert_ ( y1._node in context._correlations._mat )
        self.assert_ ( y2._node in context._correlations._mat )
        self.assert_ ( y3._node in context._correlations._mat )

        del y6
        
        # self.assert_ ( uid1 in context._dof_record )
        # self.assert_ ( uid2 in context._dof_record )
        # self.assert_ ( uid3 in context._dof_record )
        
        # self.assert_ ( y1._node in context.labels )
        # self.assert_ ( y2._node in context.labels )
        # self.assert_ ( y3._node in context.labels )

        self.assert_ ( y1._node in context._correlations._mat )
        self.assert_ ( y2._node in context._correlations._mat )
        self.assert_ ( y3._node in context._correlations._mat )

        del y3,y5

        # self.assert_ ( uid1 in context._dof_record )
        # self.assert_ ( uid2 in context._dof_record )
        # self.assert_ ( uid3 not in context._dof_record )
        
        # self.assert_ ( y1._node in context.labels )
        # self.assert_ ( y2._node in context.labels )

        self.assert_ ( y1._node in context._correlations._mat )
        self.assert_ ( y2._node in context._correlations._mat )

        del y2, y4      

        # self.assert_ ( uid1 in context._dof_record )
        # self.assert_ ( uid2 not in context._dof_record )
        
        # self.assert_ ( y1._node in context.labels )

        self.assert_ ( y1._node in context._correlations._mat )
        # self.assert_ ( uid2 not in context._correlations._mat )

        del y1

        # self.assert_ ( uid1 not in context._dof_record )
        # self.assert_ ( uid1 not in context._correlations._mat )

    def test_errors(self):
        """
        Error conditions:
##            - trying to archive a complex and one of its components, or vice versa
##            - trying to archive the same object with different names
            - trying to extract with the wrong name
            - adding to an archive that has been frozen
            - extracting from an archive that has not been thawed
            - restoring an archive to the same context
            
        """
        x = ureal(1,1)
        y = ureal(2,1)
        z = x + y

        ar = archive.Archive()

        ar.add(x=x,z=z)
##        self.assertRaises(RuntimeError,ar.add,y=x)  # same object
        self.assertRaises(RuntimeError,ar.add,x=y)  # same name


##        zc = ucomplex(1,1)
##        ar.add(zc=zc)
##        self.assertRaises(RuntimeError,ar.add,zr=zc.real)  # a component
        
##        zz = ucomplex(1,1)
##        ar.add(zr=zz.real)
##        self.assertRaises(RuntimeError,ar.add,zz=zz)  # a component was archived

        ar._freeze()
        self.assertRaises(RuntimeError,ar.extract,'x')  # cannot extract yet

##        self.assertRaises(RuntimeError,ar._thaw,default.context)  # cannot extract 
        
        ar._thaw( Context() )

        self.assertRaises(RuntimeError,ar.add,y=y)  # cannot add now
        
    def test(self):
        """
        Simple x,y z problem, but don't save
        one of the elementary uns.
        
        """
        x = ureal(1,1)
        y = ureal(2,1)
        z = archive.result( x + y )
        
        self.assert_( z._is_intermediate )

        ar = archive.Archive()

        ar.add(x=x,z=z)
        ar._freeze()
                
        # Unpack and check 
        context = Context()
        ar._thaw(context)

        # Single argument unpacks to an object
        # not a sequence
        x1 = ar.extract('x')
        z1 = ar.extract('z')

        self.assert_( z1._is_intermediate )
        self.assert_(
            equivalent_matt(
                z._i_components,z1._i_components
            ) )

        self.assertEqual(z.s,z1.s)
        self.assertEqual(x.s,x1.s)

        a = component(z,x)
        b = component(z1,x1)
        self.assert_( equivalent(a,b,TOL) )

        # The restored uncertain numbers will
        # be indistinguishable from the originals
        # in terms of components
        a = component(z1,x)
        b = component(z,x1)
        self.assert_( equivalent(a,b,TOL) )

        a = z1._node.partial_derivative(x1._node)
        self.assertEqual( a,1 )
        b = z._node.partial_derivative(x._node)
        self.assert_( equivalent(a,b,TOL) )

        # However the nodes are not the same. 
        a = z._node.partial_derivative(x1._node)
        self.assertEqual( a,0 )

        # Make sure the vector of components is well-formed
        self.assert_( is_ordered(z1._u_components) )
        
    def test_attributes(self):
        """
        Dict-like attributes: keys(), values() items()
        and the 'iter' variants, also len()
        
        """
        x = ureal(1,1)
        y = ureal(2,1)
        z = x + y

        ar = archive.Archive()
        self.assertEqual( len(ar) , 0)
        ar.add(x=x,z=z)
        self.assertEqual( len(ar) , 2)

        names = ['x','z']
        objs = [x,z]
        self.assertEqual( names,ar.keys() )
        self.assertEqual( objs,ar.values() )
        self.assertEqual( zip(names,objs),ar.items() )

        for i,k in enumerate( ar.iterkeys() ):
            self.assertEqual( names[i], k )

        for i,k in enumerate( ar.itervalues() ):
            self.assertEqual( objs[i], k )

        items = zip(names,objs)
        for i,k in enumerate( ar.iteritems() ):
            self.assertEqual( items[i], k )

    def test1(self):
        """
        Simple tests: elementary and intermediate
        unceratin numbers restored OK? Saving
        all objects.
        
        """
        x = ureal(1,1)
        y = ureal(2,1)
        z = archive.result( x + y )

        ar = archive.Archive()

        ar.add(x=x,y=y,z=z)
        ar._freeze()

        # Unpack and check 
        context = Context()
        ar._thaw(context)

        x1, y1, z1 = ar.extract('x','y','z')

        self.assertEqual(x.s,x1.s)
        self.assertEqual(y.s,y1.s)
        self.assertEqual(z.s,z1.s)

        a = component(z,x)
        b = component(z1,x1)
        self.assert_( equivalent(a,b,TOL) )

        # The restored uncertain numbers will
        # be indistinguishable from the originals
        # in terms of components
        a = component(z1,x)
        b = component(z,x1)
        self.assert_( equivalent(a,b,TOL) )

        a = z1._node.partial_derivative(x1._node)
        self.assertEqual( a,1 )
        b = z._node.partial_derivative(x._node)
        self.assert_( equivalent(a,b,TOL) )

        # However the nodes are not the same. 
        a = z._node.partial_derivative(x1._node)
        self.assertEqual( a,0 )

        # Make sure the vector of components is well-formed
        self.assert_( is_ordered(z1._u_components) )

    def test2(self):
        """
        Logical correlation works - I
        with all objects archived
        
        """
        x1 = ureal(1,1)
        x2 = ureal(1,1)
        x3 = ureal(1,1)
        x4 = archive.result( x1 + x2 )
        x5 = x2 + x3 
        x6 = archive.result( x4 + x5 )

        ar = archive.Archive()
        ar.add(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5,x6=x6)
        ar._freeze()

        # unpack and test
        context = Context()
        ar._thaw(context)
        y1, y2, y3 = ar.extract('x1','x2','x3')

        self.assertEqual(x1.s,y1.s)
        self.assertEqual(x2.s,y2.s)
        self.assertEqual(x3.s,y3.s)

        y4, y5, y6 = ar.extract('x4','x5','x6')
        
        self.assertEqual(x4.s,y4.s)
        self.assertEqual(x5.s,y5.s)
        self.assertEqual(x6.s,y6.s)

        a = component(x6,x4)
        b = component(y6,y4)
        self.assert_( equivalent(a,b,TOL) )

        a = component(x6,x2)
        b = component(y6,y2)
        self.assert_( equivalent(a,b,TOL) )

        a = get_correlation(x4,x5)
        b = get_correlation(y4,y5)
        self.assert_( equivalent(a,b,TOL) )

        # Make sure the vector of components is well-formed
        self.assert_( is_ordered(y6._u_components) )

    def test3(self):
        """
        Logical correlation works - II
        with all objects archived but not all restored
        
        """
        x1 = ureal(1,1)
        x2 = ureal(1,1)
        x3 = ureal(1,1)
        x4 = archive.result( x1 + x2 )
        x5 = x2 + x3
        x6 = archive.result( x4 + x5 )

        ar = archive.Archive()
        ar.add(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5,x6=x6)
        ar._freeze()

        # unpack without elementary uns and test
        context = Context()
        ar._thaw(context)

        y4, y5, y6 = ar.extract('x4','x5','x6')
        
        self.assertEqual(x4.s,y4.s)
        self.assertEqual(x5.s,y5.s)
        self.assertEqual(x6.s,y6.s)

        a = component(x6,x4)
        b = component(y6,y4)
        self.assert_( equivalent(a,b,TOL) )

        # Make sure the vector of components is well-formed
        self.assert_( is_ordered(y6._u_components) )

    def test3b(self):
        """
        Logical correlation works - III
        with not all objects archived or restored
        
        """
        x1 = ureal(1,1)
        x2 = ureal(1,1)
        x3 = ureal(1,1)
        x4 = archive.result( x1 + x2 )
        x5 = x2 + x3
        x6 = archive.result( x4 + x5 )

        ar = archive.Archive()
        ar.add(x4=x4,x5=x5,x6=x6)
        ar._freeze()

        # unpack without elementary uns and test
        context = Context()
        ar._thaw(context)

        y4, y5, y6 = ar.extract('x4','x5','x6')
        
        self.assertEqual(x4.s,y4.s)
        self.assertEqual(x5.s,y5.s)
        self.assertEqual(x6.s,y6.s)

        a = component(x6,x4)
        b = component(y6,y4)
        self.assert_( equivalent(a,b,TOL) )

        # Make sure the vector of components is well-formed
        self.assert_( is_ordered(y6._u_components) )

    def test4(self):
        """
        Explicit correlation works - I
        with all objects archived
        
        """
        x1 = ureal(1,1,independent=False)
        x2 = ureal(1,1,independent=False)
        x3 = ureal(1,1,independent=False)
        x4 = x1 + x2
        x5 = x2 + x3
        x6 = x4 + x5

        set_correlation(0.5,x1,x2)
        set_correlation(-0.25,x3,x2)

        ar = archive.Archive()
        ar.add(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5,x6=x6)
        ar._freeze()

        # unpack without elementary uns and test
        context = Context()
        ar._thaw(context)

        y1, y2, y3 = ar.extract('x1','x2','x3')

        a = get_correlation(x1,x2)
        b = get_correlation(y1,y2)
        self.assert_( equivalent(a,b,TOL) )

        a = get_correlation(x3,x2)
        b = get_correlation(y3,y2)
        self.assert_( equivalent(a,b,TOL) )

        y4, y5, y6 = ar.extract('x4','x5','x6')
        
        self.assertEqual(x4.s,y4.s)
        self.assertEqual(x5.s,y5.s)
        self.assertEqual(x6.s,y6.s)

        a = get_correlation(x4,x5)
        b = get_correlation(y4,y5)
        self.assert_( equivalent(a,b,TOL) )

        # Make sure the vector of components is well-formed
        self.assert_( is_ordered(y6._u_components) )

    def test5(self):
        """
        Explicit correlation works - II
        with only some objects restored
        
        """
        x1 = ureal(1,1,independent=False)
        x2 = ureal(1,1,independent=False)
        x3 = ureal(1,1,independent=False)
        x4 = x1 + x2
        x5 = x2 + x3
        x6 = x4 + x5

        set_correlation(0.5,x1,x2)
        set_correlation(-0.25,x3,x2)

        ar = archive.Archive()
        ar.add(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5,x6=x6)
        ar._freeze()

        # unpack without elementary uns and test
        context = Context()
        ar._thaw(context)

        y4, y5, y6 = ar.extract('x4','x5','x6')
        
        self.assertEqual(x4.s,y4.s)
        self.assertEqual(x5.s,y5.s)
        self.assertEqual(x6.s,y6.s)

        a = get_correlation(x4,x5)
        b = get_correlation(y4,y5)
        self.assert_( equivalent(a,b,TOL) )

        # Make sure the vector of components is well-formed
        self.assert_( is_ordered(y6._u_components) )

    def test5b(self):
        """
        Explicit correlation works - II
        with only some objects archived and restored
        
        """
        x1 = ureal(1,1,independent=False)
        x2 = ureal(1,1,independent=False)
        x3 = ureal(1,1,independent=False)
        x4 = x1 + x2
        x5 = x2 + x3
        x6 = x4 + x5

        set_correlation(0.5,x1,x2)
        set_correlation(-0.25,x3,x2)

        ar = archive.Archive()
        ar.add(x4=x4,x5=x5,x6=x6)
        ar._freeze()

        # unpack without elementary uns and test
        context = Context()
        ar._thaw(context)

        y4, y5, y6 = ar.extract('x4','x5','x6')
        
        self.assertEqual(x4.s,y4.s)
        self.assertEqual(x5.s,y5.s)
        self.assertEqual(x6.s,y6.s)

        a = get_correlation(x4,x5)
        b = get_correlation(y4,y5)
        self.assert_( equivalent(a,b,TOL) )
        
        # Make sure the vector of components is well-formed
        self.assert_( is_ordered(y6._u_components) )

    def test6(self):
        """
        Same as test 5 but with [] indexing
        
        """
        x1 = ureal(1,1,independent=False)
        x2 = ureal(1,1,independent=False)
        x3 = ureal(1,1,independent=False)
        x4 = x1 + x2
        x5 = x2 + x3
        x6 = x4 + x5

        set_correlation(0.5,x1,x2)
        set_correlation(-0.25,x3,x2)

        ar = archive.Archive()
        ar['x1'] = x1
        ar['x2'] = x2
        ar['x3'] = x3
        ar['x4'] = x4
        ar['x5'] = x5
        ar['x6'] = x6
        ar._freeze()

        # unpack without elementary uns and test
        context = Context()
        ar._thaw(context)

        y4 = ar['x4']
        y5 = ar['x5']
        y6 = ar['x6']
        
        self.assertEqual(x4.s,y4.s)
        self.assertEqual(x5.s,y5.s)
        self.assertEqual(x6.s,y6.s)

        a = get_correlation(x4,x5)
        b = get_correlation(y4,y5)
        self.assert_( equivalent(a,b,TOL) )

        # Make sure the vector of components is well-formed
        self.assert_( is_ordered(y6._u_components) )

    def testGUMH2(self):
        """
        GUM H2 as an ensemble
        
        """
        x = [4.999,0.019661,1.04446]
        u = [0.0032,0.0000095,0.00075]
        
        v,i,phi = multiple_ureal(x,u,5)
        
        set_correlation(-0.36,v,i)
        set_correlation(0.86,v,phi)
        set_correlation(-0.65,i,phi)

        r = v * cos(phi)/ i
        x = v * sin(phi)/ i
        z = v / i

        ar = archive.Archive()
        ar.add(v=v,i=i,phi=phi)
        ar._freeze()

        # unpack without elementary uns and test
        context = Context()
        ar._thaw(context)

        v1, i1, phi1 = ar.extract('v','i','phi')

        self.assert_( v1._node in context._ensemble )
        self.assert_( i1._node in context._ensemble )
        self.assert_( phi1._node in context._ensemble )
        
        self.assertEqual(v1.s,v.s)
        self.assertEqual(i1.s,i.s)
        self.assertEqual(phi1.s,phi.s)
               
        r1 = v1 * cos(phi1)/ i1
        x1 = v1 * sin(phi1)/ i1
        z1 = v1 / i1

        # The DoF calculation would fail if the inputs
        # are not part of the same ensemble.    
        self.assert_( equivalent( dof(r1),5,TOL) )
        self.assert_( equivalent( dof(x1),5,TOL) )
        self.assert_( equivalent( dof(z1),5,TOL) )

        self.assert_( equivalent( value(r1),value(r),TOL) )
        self.assert_( equivalent( value(x1),value(x),TOL) )
        self.assert_( equivalent( value(z1),value(z),TOL) )

        self.assert_( equivalent( uncertainty(r1),uncertainty(r),TOL) )
        self.assert_( equivalent( uncertainty(x1),uncertainty(x),TOL) )
        self.assert_( equivalent( uncertainty(z1),uncertainty(z),TOL) )

        # Make sure the vector of components is well-formed
        self.assert_( is_ordered(r1._u_components) )
        self.assert_( is_ordered(x1._u_components) )
        self.assert_( is_ordered(z1._u_components) )

    def testGUMH2_b(self):
        """
        GUM H2 as an ensemble. Only save the results this time.
        
        """
        x = [4.999,0.019661,1.04446]
        u = [0.0032,0.0000095,0.00075]
        
        v,i,phi = multiple_ureal(x,u,5)
        
        set_correlation(-0.36,v,i)
        set_correlation(0.86,v,phi)
        set_correlation(-0.65,i,phi)

        r = v * cos(phi)/ i
        x = v * sin(phi)/ i
        z = v / i

        ar = archive.Archive()
        ar.add(r=r,x=x,z=z)
        ar._freeze()

        # unpack without elementary uns and test
        context = Context()
        ar._thaw(context)

        r1, x1, z1 = ar.extract('r','x','z')
        
        self.assertEqual(r1.s,r.s)
        self.assertEqual(x1.s,x.s)
        self.assertEqual(z1.s,z.s)
               
        # The DoF calculation would fail if the inputs
        # are not part of the same ensemble.    
        self.assert_( equivalent( dof(r1),5,TOL) )
        self.assert_( equivalent( dof(x1),5,TOL) )
        self.assert_( equivalent( dof(z1),5,TOL) )

        self.assert_( equivalent( value(r1),value(r),TOL) )
        self.assert_( equivalent( value(x1),value(x),TOL) )
        self.assert_( equivalent( value(z1),value(z),TOL) )

        self.assert_( equivalent( uncertainty(r1),uncertainty(r),TOL) )
        self.assert_( equivalent( uncertainty(x1),uncertainty(x),TOL) )
        self.assert_( equivalent( uncertainty(z1),uncertainty(z),TOL) )
        
        # Make sure the vector of components is well-formed
        self.assert_( is_ordered(r1._u_components) )
        self.assert_( is_ordered(x1._u_components) )
        self.assert_( is_ordered(z1._u_components) )

    def test_intermediates_multipath(self):
        """
        Check the sensitivities when the tree structure
        has intermediates that depend on intermediates

        In this case y is directly connected to x1, x4 and x5
        but x4 and x5 also depend on x1 and x5 depends on x4

        The archiving algorithm must  
        """
        x1 = ureal(1.2,0.6,label='x1')
        x2 = ureal(2.5,1.6,label='x2')
        x3 = ureal(-5.3,.77,label='x3')
        x4 = archive.result( 2.0 * (x1 + x2) )
        x5 = archive.result( (x4 + x3)**2 )
        y = archive.result( sqrt(x5 + x4 + x1) )

        ar = archive.Archive()
        ar.add(y=y,x4=x4,x5=x5,x1=x1)
        ar._freeze()

        # unpack without elementary uns and test
        context = Context()
        ar._thaw(context)
        yy, xx4, xx5, xx1 = ar.extract('y','x4','x5','x1')

        self.assert_(
            equivalent(
                rp.sensitivity(y,x4),
                rp.sensitivity(yy,xx4),
                TOL
            )
        )

        self.assert_(
            equivalent(
                rp.sensitivity(y,x5),
                rp.sensitivity(yy,xx5),
                TOL
            )
        )

        self.assert_(
            equivalent(
                rp.sensitivity(y,x1),
                rp.sensitivity(yy,xx1),
                TOL
            )
        )

        self.assert_(
            equivalent(
                rp.sensitivity(x5,x1),
                rp.sensitivity(xx5,xx1),
                TOL
            )
        )

        self.assert_(
            equivalent(
                rp.sensitivity(x4,x1),
                rp.sensitivity(xx4,xx1),
                TOL
            )
        )

        self.assert_(
            equivalent(
                rp.sensitivity(x5,x4),
                rp.sensitivity(xx5,xx4),
                TOL
            )
        )

        self.assert_(
            equivalent(
                rp.sensitivity(x4,x5),
                rp.sensitivity(xx4,xx5),
                TOL
            )
        )
     
    def test_complex_1(self):
        """
        Simple arithmetic - all objects stored
        and restored.
        
        """
        x = ucomplex(1,[1,2],4)
        y = ucomplex(1,[3,12],3)
        z = x * y

        ar = archive.Archive()
        ar.add(x=x,y=y,z=z)
        ar._freeze()

        # unpack 
        context = Context()
        ar._thaw(context)

        x1, y1, z1 = ar.extract('x','y','z')

        self.assertEqual(x1.s,x.s)
        self.assertEqual(y1.s,y.s)

        self.assert_( equivalent_complex(x1.x,x.x) )
        self.assert_( equivalent_sequence(x1.u,x.u) )
        self.assert_( equivalent(x1.df,x.df) )

        self.assert_( equivalent_complex(y1.x,y.x) )
        self.assert_( equivalent_sequence(y1.u,y.u) )
        self.assert_( equivalent(y1.df,y.df) )
        
        self.assert_( equivalent_complex(z1.x,z.x) )
        self.assert_( equivalent_sequence(z1.u,z.u) )
        self.assert_( equivalent(z1.df,z.df) )

        # Make sure the vector of components is well-formed
        self.assert_( is_ordered(y1.real._u_components) )
        self.assert_( is_ordered(x1.real._u_components) )
        self.assert_( is_ordered(z1.real._u_components) )
        self.assert_( is_ordered(y1.imag._u_components) )
        self.assert_( is_ordered(x1.imag._u_components) )
        self.assert_( is_ordered(z1.imag._u_components) )

    def test_complex_2(self):
        """
        Complex logical correlation
        
        """
        x1 = ucomplex(1,[3,2],4)
        x2 = ucomplex(1,[1,1],5)
        x3 = ucomplex(1,[4,5],6)

        x4 = x1*x2
        x5 = x2*x3

        x6 = x4 + x5

        ar = archive.Archive()
        ar.add(x4=x4,x5=x5,x6=x6)
        ar._freeze()

        ar._thaw( Context() )
        y4, y5, y6 = ar.extract('x4','x5','x6')

        self.assertEqual(x4.s,y4.s)
        self.assertEqual(x5.s,y5.s)
        self.assertEqual(x6.s,y6.s)

        self.assert_( equivalent_complex(x4.x,y4.x) )
        self.assert_( equivalent_sequence(x4.u,y4.u) )
        self.assert_( equivalent(x4.df,y4.df) )

        self.assert_( equivalent_complex(x5.x,y5.x) )
        self.assert_( equivalent_sequence(x5.u,y5.u) )
        self.assert_( equivalent(x5.df,y5.df) )
        
        self.assert_( equivalent_complex(x6.x,y6.x) )
        self.assert_( equivalent_sequence(x6.u,y6.u) )
        self.assert_( equivalent(x6.df,y6.df) )

        self.assert_( equivalent_sequence(
            get_correlation(x6,x5),
            get_correlation(y6,y5)
        ) )
                               
        self.assert_( equivalent_sequence(
            get_correlation(x6,x4),
            get_correlation(y6,y4)
        ) )

        # Make sure the vector of components is well-formed
        self.assert_( is_ordered(x6.real._u_components) )
        self.assert_( is_ordered(x6.imag._u_components) )

    def test_complex_3(self):
        """
        Complex logical and external correlation
        
        """
        x1 = ucomplex(1,[3,2],independent=False)
        x2 = ucomplex(1,[1,1],independent=False)
        x3 = ucomplex(1,[4,5])

        R = [.1,.2,.3,.4]
        set_correlation(R,x1,x2)
        # Make sure we get what we expect
        self.assert_( equivalent_sequence(
            R, get_correlation(x1,x2)
        ))
        
        x4 = x1*x2
        x5 = x2*x3

        x6 = x4 + x5

        ar = archive.Archive()
        ar.add(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5,x6=x6)
        ar._freeze()

        ar._thaw( Context() )
        y4, y5, y6 = ar.extract('x4','x5','x6')

        self.assertEqual(x4.s,y4.s)
        self.assertEqual(x5.s,y5.s)
        self.assertEqual(x6.s,y6.s)

        self.assert_( equivalent_sequence(
            get_correlation(x6,x5),
            get_correlation(y6,y5)
        ) )
                               
        self.assert_( equivalent_sequence(
            get_correlation(x6,x4),
            get_correlation(y6,y4)
        ) )

        self.assert_( equivalent_sequence(
            variance(x6),
            variance(y6)
        ))
        
        # Now look at the elementary influences
        y1, y2 = ar.extract('x1','x2')
        self.assert_( equivalent_sequence(
            R, get_correlation(y1,y2)
        ))

        # Make sure the vector of components is well-formed
        self.assert_( is_ordered(x6.real._u_components) )
        self.assert_( is_ordered(x6.imag._u_components) )

    def test_complex_4(self):
        """
        Same as 'complex_3' but using [] indexing
        
        """
        x1 = ucomplex(1,[3,2],independent=False)
        x2 = ucomplex(1,[1,1],independent=False)
        x3 = ucomplex(1,[4,5])

        R = [.1,.2,.3,.4]
        set_correlation(R,x1,x2)
        # Make sure we get what we expect
        self.assert_( equivalent_sequence(
            R, get_correlation(x1,x2)
        ))
        
        x4 = x1*x2
        x5 = x2*x3

        x6 = x4 + x5

        ar = archive.Archive()
        ar['x1'] = x1 
        ar['x2'] = x2 
        ar['x3'] = x3 
        ar['x4'] = x4 
        ar['x5'] = x5 
        ar['x6'] = x6 
        ar._freeze()

        ar._thaw( Context() )
        y4 = ar['x4']
        y5 = ar['x5']
        y6 = ar['x6']

        self.assertEqual(x4.s,y4.s)
        self.assertEqual(x5.s,y5.s)
        self.assertEqual(x6.s,y6.s)

        self.assert_( equivalent_sequence(
            get_correlation(x6,x5),
            get_correlation(y6,y5)
        ) )
                               
        self.assert_( equivalent_sequence(
            get_correlation(x6,x4),
            get_correlation(y6,y4)
        ) )

        self.assert_( equivalent_sequence(
            variance(x6),
            variance(y6)
        ))
        
        # Now look at the elementary influences
        y1, y2 = ar.extract('x1','x2')
        self.assert_( equivalent_sequence(
            R, get_correlation(y1,y2)
        ))

        # Make sure the vector of components is well-formed
        self.assert_( is_ordered(x6.real._u_components) )
        self.assert_( is_ordered(x6.imag._u_components) )

    def test_complex_5(self):
        """
        Archive intermediate complex components
        
        """
        x1 = ucomplex(1,[3,2],independent=False)
        x2 = ucomplex(1,[1,1],independent=False)
        x3 = ucomplex(1,[4,5])

        R = [.1,.2,.3,.4]
        set_correlation(R,x1,x2)
        # Make sure we get what we expect
        self.assert_( equivalent_sequence(
            R, get_correlation(x1,x2)
        ))
        
        x4 = archive.result( x1*x2 )
        x5 = archive.result( x2*x3 )

        x6 = archive.result( x4 + x5  )
        x7 = archive.result( mag_squared(x6) + mag_squared(x4) )

        ar = archive.Archive()
        ar['x1'] = x1 
        ar['x2'] = x2 
        ar['x3'] = x3 
        ar['x4'] = x4.real  # just the real component
        ar['x5'] = x5 
        ar['x6'] = x6
        ar['x7'] = x7
        ar._freeze()

        ar._thaw( Context() )
        y4 = ar['x4']
        y5 = ar['x5']
        y6 = ar['x6']
        y7 = ar['x7']

        self.assertEqual(x4.real.s,y4.s)
        self.assertEqual(x5.s,y5.s)
        self.assertEqual(x6.s,y6.s)

        # Archived as a complex influence of a complex quantity                
        self.assert_( equivalent_sequence(
            rp.u_component(x6,x5),
            rp.u_component(y6,y5)
        ) )

        # Archived as a real influence of a complex quantity                
        self.assert_( equivalent_sequence(
            rp.u_component(x6,x4.real),
            rp.u_component(y6,y4)
        ) )

        # Archived as a complex influence of a real quantity                
        self.assert_( equivalent_sequence(
            rp.u_component(x7,x5),
            rp.u_component(y7,y5)
        ))
        
        # Archived as a real influence of a real quantity                
        self.assert_( equivalent(
            rp.u_component(x7,x4.real),
            rp.u_component(y7,y4)
        ))

    def test_nop_intermediates(self):
        """
        Archiving irrelevant intermediate components
        
        """
        x1 = ucomplex(1,[3,2])
        x2 = ureal(1,1)
        x3 = constant(1.3)
        
        x4 = x1*x2
        x5 = x2*x3

        x6 = archive.result( x4 + x5 )
        x7 = archive.result( mag_squared(x6) )

        ar = archive.Archive()
        ar.add(
            x1=x1,x2=x2,x3=x3,x4=x4.real,
            x5=x5,x6=x6,x7=x7
        )
        ar._freeze()

        ar._thaw( Context() )
        y1 = ar['x1']
        y2 = ar['x2']
        y3 = ar['x3']
        y6 = ar['x6']
        y7 = result( ar['x7'] )

        self.assertEqual(x1.s,y1.s)
        self.assertEqual(x2.s,y2.s)
        self.assertEqual(x3.s,y3.s)
        self.assertEqual(x6.s,y6.s)
        self.assertEqual(x7.s,y7.s)

        # Archived as a complex influence of a complex quantity                
        self.assert_( equivalent_sequence(
            rp.u_component(x6,x1),
            rp.u_component(y6,y1)
        ) )

        # Archived as a real influence of a complex quantity                
        self.assert_( equivalent_sequence(
            rp.u_component(x6,x2),
            rp.u_component(y6,y2)
        ) )

        # Archived as a complex influence of a real quantity                
        self.assert_( equivalent_sequence(
            rp.u_component(x7,x1),
            rp.u_component(y7,y1)
        ))
        
        # Archived as a real influence of a real quantity                
        self.assert_( equivalent(
            rp.u_component(x7,x2),
            rp.u_component(y7,y2)
        ))

        # Archived as a constant influence of a real quantity 
        self.assert_( equivalent(
            rp.u_component(x7,x3),
            rp.u_component(y7,y3)
        ))

        # Archived as a constant influence of a complex quantity                
        self.assert_( equivalent_sequence(
            rp.u_component(x6,x3),
            rp.u_component(y6,y3)
        ) )
                         
    def test_with_file(self):
        """
        Save to a file and then restore by reading
        """
        wdir = os.getcwd()
        fname = 'test_file.pkl'
        path = os.path.join(wdir,fname)

        x = ureal(1,1)
        y = ureal(2,1)
        z = x + y

        ar = archive.Archive()

        ar.add(x=x,y=y,z=z)

        f = open(path,'wb')
        archive.dump(f,ar)
        f.close()

        default.context = Context()
        f = open(path,'rb')
        ar = archive.load(f)
        f.close()
        os.remove(path)

        x1, y1, z1 = ar.extract('x','y','z')

        self.assertEqual(x.s,x1.s)
        self.assertEqual(y.s,y1.s)
        self.assertEqual(z.s,z1.s)

    def test_with_file2(self):
        """
        Save to a file and then restore several times
        to test the effectiveness of GTC's uid system.
        
        """
        wdir = os.getcwd()
        fname = 'test_file.pkl'
        path = os.path.join(wdir,fname)

        x = ureal(1,1,3,label='x')
        y = ureal(2,1,4,label='y')
        z = x + y

        ar = archive.Archive()

        # Saving only `z` means that when the archive
        # is restored `x` and `y` are not recreated within
        # the archive. This creates a problem, because if
        # the archive is restored more than once in the same
        # context there could be different LeafNodes associated
        # with these elementary UNs. If one of these LNs is
        # garbage collected it signals the context causing DoF and
        # label info to be removed from the registers.
        # A cache has been introduced to make sure that restored
        # leafNodes are unique. This test verifies that.
        
        ar.add(z=z)

        with open(path,'wb') as f:
            archive.dump(f,ar)

        default.context = Context()
        
        with open(path,'rb') as f:
            ar1 = archive.load(f)

        z1 = ar1.extract('z')

        with open(path,'rb') as f:
            ar2 = archive.load(f)

        z2 = ar2.extract('z')

        # z1 and z2 should be indistinguishable
        self.assertEqual(z1.s,z2.s)
        self.assertEqual(get_correlation(z1,z2),1)
        budgets = zip( rp.budget(z1,trim=0), rp.budget(z2,trim=0) )
        for c1,c2 in budgets:
            self.assertEqual(c1[0],c2[0])
            self.assert_( equivalent(c1[1],c2[1]) )

        # The contexts are different so correlation between
        # z and z2 will not be unity, however
        self.assertEqual(z.s,z2.s)
        budgets = zip( rp.budget(z,trim=0), rp.budget(z2,trim=0) )
        for c1,c2 in budgets:
            self.assertEqual(c1[0],c2[0])
            self.assert_( equivalent(c1[1],c2[1]) )
          
        # Deleting `z1` will cause it's node tree to collapse
        # However, we require the leaves of that tree to endure
        # because `z2` is still alive.
        del z1
        
        # Testing the summary strings against `z` ensures that
        # the DoF information is still accessible to `z2`
        # (otherwise dof would be infinity).
        # Comparing the budgets ensures that the influence labels
        # have not been removed.
        # The removal of dof and label information would happen
        # if the `z1` tree LeafNodes were garbage collected, calling
        # the `deregister` context method.
        self.assertEqual(z.s,z2.s)
        budgets = zip( rp.budget(z,trim=0), rp.budget(z2,trim=0) )
        for c1,c2 in budgets:
            self.assertEqual(c1[0],c2[0])
            self.assert_( equivalent(c1[1],c2[1]) )

        os.remove(path)
 
    def test_with_string(self):
        """
        Save to a file and then restore by reading
        """
        x = ureal(1,1)
        y = ureal(2,1)
        z = x + y

        ar = archive.Archive()

        ar.add(x=x,y=y,z=z)

        db = archive.dumps(ar)

        default.context = Context()
        ar = archive.loads(db)

        x1, y1, z1 = ar.extract('x','y','z')

        self.assertEqual(x.s,x1.s)
        self.assertEqual(y.s,y1.s)
        self.assertEqual(z.s,z1.s)

    def test_multiple_names(self):
        """
        Different names may apply to the same UN
        
        """
        x1 = ureal(1,1)
        x2 = ureal(1,1)
        x3 = ureal(1,1)
        x4 = archive.result( x1 + x2 )
        x5 = x2 + x3
        x6 = archive.result( x4 + x5 )

        ar = archive.Archive()
        ar.add(
            x1=x1,x2=x2,x3=x3,x4=x4,x5=x5,x6=x6,
            z1=x1,z4=x4,z6=x6
        )
        ar._freeze()

        # unpack and test
        context = Context()
        ar._thaw(context)
        y1, y2, y3 = ar.extract('x1','x2','x3')
        yz1 = ar['z1']
        self.assert_( yz1 is y1 )
        self.assert_( yz1._node is y1._node )
        self.assertEqual(x1.s,y1.s)
        self.assertEqual(yz1.s,y1.s)

        # NB most of these tests are redundant
        # now that multiple names are restored
        # to the same object. 
        y4, y5, y6 = ar.extract('x4','x5','x6')
        yz4 = ar['z4']
        yz6 = ar['z6']
        self.assert_( yz4 is y4 )
        self.assert_( yz4._node is y4._node )
        self.assert_( yz6 is y6 )
        self.assert_( yz6._node is y6._node )
        
        self.assertEqual(x4.s,y4.s)
        self.assertEqual(yz4.s,y4.s)

        self.assertEqual(x6.s,y6.s)
        self.assertEqual(yz6.s,y6.s)

        a = component(x6,x4)
        b = component(y6,y4)
        self.assert_( equivalent(a,b,TOL) )

        b = component(yz6,y4)
        self.assert_( equivalent(a,b,TOL) )

        b = component(yz6,yz4)
        self.assert_( equivalent(a,b,TOL) )

        # Make sure the vector of components is well-formed
        self.assert_( is_ordered(yz6._u_components) )
        
    def test_archive_with_previously_archived(self):
        """
        Make sure that an archive can be created that
        includes some UNs that have been restored from
        another archive.

        NB some tests here imposed interchangeability of
        objects that belong to different contexts. But
        this is not a requirement, so I have commented
        out. We should never have to work with a mixture 
        of uncertain number instances created by different
        Contexts.
        
        """
        # Archive some simple stuff
        # This is the first archive. 
        x1 = ureal(1,1,5,label='x1')
        x2 = ucomplex(1,1,7,label='x2')
        x3 = archive.result( x1 + x2 )
        self.assert_( x3.real._is_intermediate )   

        ar = archive.Archive()
        ar.add(x1=x1,x2=x2,x3=x3)
        ar._freeze()

        # Now add some more stuff and store
        # in a second archive
        default.context = Context()
        
        ar._thaw(default.context)
        y1 = ar['x1']
        y2 = ar['x2']
        y3 = ar['x3'] 

        self.assert_( y1._is_elementary )
        self.assert_( y2._is_elementary )
        self.assert_( not y3._is_elementary ) 

        self.assertEqual(y3.real._uid,x3.real._uid)
        self.assert_( y3.real._is_intermediate )   

        self.assert_(
            equivalent(component(x3,x1),component(y3,y1))
        )
        self.assert_(
            equivalent_matt(x3.real._i_components,y3.real._i_components)
        )
##        # This test fails because x._uid != y._uid (the
##        # restored object has the full context code)
##        self.assert_(
##            equivalent(component(x3,x1),component(y3,x1))
##        )

        y4 = ureal(10,1,23)
        y5 = ucomplex(2.5+7j,2,19)
        y6 = archive.result( y3 * (y4 + y5) )
        
        ar = archive.Archive()
        ar.add(
            y1=y1,
            y2=y2,
            y3=y3,
            y4=y4,y5=y5,y6=y6
        )
        ar._freeze()

        # Now retore into a third Context
        default.context = Context()

        ar._thaw(default.context)
        z1 = ar['y1']
        z2 = ar['y2']
        z3 = ar['y3']

        self.assert_( z1._is_elementary )
        self.assert_( z2._is_elementary )
        self.assert_( not z3._is_elementary )   

        # First just check that we get the same objects
        # and that the values are correct
        self.assertEqual(z1._uid,y1._uid)
        self.assertEqual(z1._uid[1],x1._uid[1])
##        self.assertEqual(z1._uid[0],x1._uid[0])

        self.assert_(
            equivalent(value(z1),value(y1))
        )
        self.assert_(
            equivalent(value(z1),value(x1))
        )
        self.assert_(
            equivalent(uncertainty(z1),uncertainty(y1))
        )
        self.assert_(
            equivalent(uncertainty(z1),uncertainty(x1))
        )
        self.assert_(
            equivalent(dof(z1),dof(y1))
        )
        self.assert_(
            equivalent(dof(z1),dof(x1))
        )

        self.assertEqual(z2._uid,y2._uid)
        self.assertEqual(z2.real._uid,y2.real._uid)
        self.assertEqual(z2.real._uid[1],x2.real._uid[1])
##        self.assertEqual(z2.real._uid[0],x2.real._uid[0])
        self.assert_(
            equivalent(value(z2.real),value(y2.real))
        )
        self.assert_(
            equivalent(value(z2.real),value(x2.real))
        )
        self.assert_(
            equivalent(uncertainty(z2.real),uncertainty(y2.real))
        )
        self.assert_(
            equivalent(uncertainty(z2.real),uncertainty(x2.real))
        )
        self.assert_(
            equivalent(dof(z2),dof(y2))
        )
        self.assert_(
            equivalent(dof(z2),dof(x2))
        )
        self.assertEqual(z2.imag._uid,y2.imag._uid)
        self.assertEqual(z2.imag._uid[1],x2.imag._uid[1])
##        self.assertEqual(z2.imag._uid[0],x2.imag._uid[0])
        
        self.assert_(
            equivalent(value(z2.imag),value(y2.imag))
        )
        self.assert_(
            equivalent(value(z2.imag),value(x2.imag))
        )
        self.assert_(
            equivalent(uncertainty(z2.imag),uncertainty(y2.imag))
        )
        self.assert_(
            equivalent(uncertainty(z2.imag),uncertainty(x2.imag))
        )

        self.assert_(
            equivalent(value(z3.imag),value(y3.imag))
        )
        self.assert_(
            equivalent(value(z3.imag),value(x3.imag))
        )
        self.assert_(
            equivalent(uncertainty(z3.imag),uncertainty(y3.imag))
        )
        self.assert_(
            equivalent(uncertainty(z3.imag),uncertainty(x3.imag))
        )

        # Uncertainty components
        self.assert_(
            equivalent(component(x3,x1),component(z3,z1))
        )
##        self.assert_(
##            equivalent(component(x3,x1),component(z3,y1))
##        )
##        # This test fails because x._uid != y._uid (the
##        # restored object has the full context code)
##        self.assert_(
##            equivalent(component(x3,x1),component(z3,x1))
##        )
        self.assert_(
            equivalent_sequence(rp.u_component(x3,x2),rp.u_component(z3,z2))
        )
##        self.assert_(
##            equivalent_sequence(rp.u_component(x3,x2),rp.u_component(z3,y2))
##        )
##        # This test fails because x._uid != y._uid (the
##        # restored object has the full context code)
##        self.assert_(
##            equivalent_sequence(rp.u_component(x3,x2),rp.u_component(z3,x2))
##        )

        # Continue to restore
        z4 = ar['y4']
        z5 = ar['y5']
        z6 = ar['y6']
        
        # Make sure the indexing of components is still ordered
        self.assert_( is_ordered(z6.real._u_components) )

        # First just check that we get the same objects
        # and that the values are correct
        self.assertEqual(z4._uid[1],y4._uid[1])
##        self.assertEqual(z4._uid[0],y4._uid[0])
        self.assert_(
            equivalent(value(z4),value(y4))
        )
        self.assert_(
            equivalent(uncertainty(z4),uncertainty(y4))
        )
        self.assert_(
            equivalent(dof(z4),dof(y4))
        )

        self.assertEqual(z5.real._uid[1],y5.real._uid[1])
##        self.assertEqual(z5.real._uid[0],y5.real._uid[0])
        self.assert_(
            equivalent(value(z5.real),value(y5.real))
        )
        self.assert_(
            equivalent(uncertainty(z5.real),uncertainty(y5.real))
        )
        self.assert_(
            equivalent(uncertainty(z5.imag),uncertainty(y5.imag))
        )

        # Components
        # We will get components wrt elementary inputs
        self.assert_(
            equivalent_sequence(rp.u_component(z6,z2),rp.u_component(y6,y2))
        )
        # And intermediates (provided this was requested)
        self.assert_(
            equivalent_sequence(rp.u_component(z6,z3),rp.u_component(y6,y3))
        )
##        # But not wrt intermediates defined in another context because
##        # intermediate PDs are buffered using node objects as keys.
##        # There is no obvious need to worry about this
##        self.assert_(
##            equivalent_sequence(rp.u_component(z6,z3),rp.u_component(y6,z3))
##        )

        
#============================================================================
if(__name__== '__main__'):

    unittest.main()    # Runs all test methods starting with 'test'