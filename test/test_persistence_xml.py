import io
import os
import tempfile
import unittest
from math import isinf

from lxml import etree

from GTC import *
from GTC import context
from GTC.context import Context
from testing_tools import *

schema_file = r'../GTC/schema/gtc_v_1_5_0.xsd'
_file = os.path.join(os.path.dirname(__file__), schema_file)
schema = etree.XMLSchema(file=_file)


class TestArchiveXML(unittest.TestCase):

    def test_with_file(self):
        """
        Save to a file and then restore by reading
        """
        wdir = os.getcwd()
        fname = 'test_file.xml'
        path = os.path.join(wdir, fname)

        context._context = Context()
        x = ureal(1, 1)
        y = ureal(2, 1)
        z = result(x + y)

        ar = persistence.Archive()

        ar.add(x=x, y=y, z=z)

        with open(path, 'wb') as f:
            persistence.dump_xml(f, ar)

        context._context = Context()
        with open(path, 'r') as f:
            ar = persistence.load_xml(f)
        os.remove(path)

        x1, y1, z1 = ar.extract('x', 'y', 'z')

        self.assertEqual(repr(x1), repr(x))
        self.assertEqual(repr(y1), repr(y))
        self.assertEqual(repr(z1), repr(z))

    def test_with_file2(self):
        """
        Save to a file and then restore several times
        to test the effectiveness of GTC's uid system.

        """
        wdir = os.getcwd()
        fname = 'test_file.xml'
        path = os.path.join(wdir, fname)

        context._context = Context()

        x = ureal(1, 1, 3, label='x')
        y = ureal(2, 1, 4, label='y')
        z = result(x + y)

        ar = persistence.Archive()

        # Saving only `z` means that when the archive
        # is restored `x` and `y` are not recreated as UNs.
        # However, Leaf nodes are created. We need to make sure
        # that only one Leaf node gets created.

        ar.add(z=z)

        with open(path, 'wb') as f:
            persistence.dump_xml(f, ar)

        context._context = Context()
        with open(path, 'r') as f:
            ar1 = persistence.load_xml(f)
        z1 = ar1.extract('z')

        self.assertEqual(repr(z1), repr(z))

        with open(path, 'r') as f:
            # The attempt to create the uncertain number again is allowed
            # but should not create a new node object
            ar2 = persistence.load_xml(f)
        os.remove(path)

        z2 = ar2.extract('z')
        self.assertTrue(z2.is_intermediate)
        self.assertTrue(z1._node is z2._node)

    def test_with_file3(self):
        """
        Dependent elementary UNs
        """
        wdir = os.getcwd()
        fname = 'test_file.xml'
        path = os.path.join(wdir, fname)

        context._context = Context()

        x = ureal(1, 1, independent=False)
        y = ureal(2, 1, independent=False)

        r = 0.5
        set_correlation(r, x, y)

        z = result(x + y)

        ar = persistence.Archive()

        ar.add(x=x, y=y, z=z)

        with open(path, 'wb') as f:
            persistence.dump_xml(f, ar)

        context._context = Context()
        with open(path, 'r') as f:
            ar = persistence.load_xml(f)
        os.remove(path)

        x1, y1, z1 = ar.extract('x', 'y', 'z')

        self.assertEqual(repr(x1), repr(x))
        self.assertEqual(repr(y1), repr(y))
        self.assertEqual(repr(z1), repr(z))

        self.assertEqual(get_correlation(x, y), r)

    def test_with_file4(self):
        """
        Correlations with finite DoF
        """
        wdir = os.getcwd()
        fname = 'test_file.xml'
        path = os.path.join(wdir, fname)

        context._context = Context()

        x, y = multiple_ureal([1, 2], [1, 1], 4)

        r = 0.5
        set_correlation(r, x, y)

        z = result(x + y)

        ar = persistence.Archive()

        ar.add(x=x, y=y, z=z)

        with open(path, 'wb') as f:
            persistence.dump_xml(f, ar)

        context._context = Context()
        with open(path, 'r') as f:
            ar = persistence.load_xml(f)
        os.remove(path)

        x1, y1, z1 = ar.extract('x', 'y', 'z')

        self.assertEqual(repr(x1), repr(x))
        self.assertEqual(repr(y1), repr(y))
        self.assertEqual(repr(z1), repr(z))

        self.assertEqual(get_correlation(x1, y1), r)

    def test_with_file5(self):
        """
        Complex
        """
        wdir = os.getcwd()
        fname = 'test_file.xml'
        path = os.path.join(wdir, fname)

        context._context = Context()

        x = ucomplex(1, [10, 2, 2, 10], 5)
        y = ucomplex(1 - 6j, [10, 2, 2, 10], 7)

        z = result(log(x * y))

        ar = persistence.Archive()

        ar.add(x=x, y=y, z=z)

        with open(path, 'wb') as f:
            persistence.dump_xml(f, ar)

        context._context = Context()
        with open(path, 'r') as f:
            ar = persistence.load_xml(f)
        os.remove(path)

        x1, y1, z1 = ar.extract('x', 'y', 'z')

        self.assertEqual(repr(x1), repr(x))
        self.assertEqual(repr(y1), repr(y))
        self.assertEqual(repr(z1), repr(z))

    def test_with_string1(self):
        """
        Simple save with intermediate
        """
        context._context = Context()

        x = ureal(1, 1)
        y = ureal(2, 1)
        z = result(x + y)

        ar = persistence.Archive()

        ar.add(x=x, y=y, z=z)

        db = persistence.dumps_xml(ar)

        context._context = Context()
        ar = persistence.loads_xml(db)

        x1, y1, z1 = ar.extract('x', 'y', 'z')

        self.assertEqual(repr(x1), repr(x))
        self.assertEqual(repr(y1), repr(y))
        self.assertEqual(repr(z1), repr(z))

    def test_with_string2(self):
        """
        Save to a file and then restore several times
        to test the effectiveness of GTC's uid system.

        """
        context._context = Context()

        x = ureal(1, 1, 3, label='x')
        y = ureal(2, 1, 4, label='y')
        z = result(x + y)

        ar = persistence.Archive()

        # Saving only `z` means that when the archive
        # is restored `x` and `y` are not recreated as UNs.
        # However, Leaf nodes are created. We need to make sure
        # that only one Leaf node gets created.

        ar.add(z=z)

        s = persistence.dumps_xml(ar)

        context._context = Context()

        ar1 = persistence.loads_xml(s)
        z1 = ar1.extract('z')

        # The attempt to create a new uncertain number
        # is allowed but a new node is not created
        ar2 = persistence.loads_xml(s)
        z2 = ar2.extract('z')
        self.assertTrue(z2.is_intermediate)
        self.assertTrue(z2._node is z1._node)

    def test_with_string3(self):
        """
        Dependent elementary UNs
        """
        context._context = Context()

        x = ureal(1, 1, independent=False)
        y = ureal(2, 1, independent=False)

        r = 0.5
        set_correlation(r, x, y)

        z = result(x + y)

        ar = persistence.Archive()

        ar.add(x=x, y=y, z=z)

        db = persistence.dumps_xml(ar)

        context._context = Context()
        ar = persistence.loads_xml(db)

        x1, y1, z1 = ar.extract('x', 'y', 'z')

        self.assertEqual(repr(x1), repr(x))
        self.assertEqual(repr(y1), repr(y))
        self.assertEqual(repr(z1), repr(z))

        self.assertEqual(get_correlation(x, y), r)

    def test_with_string4(self):
        """
        Correlations with finite DoF
        """
        context._context = Context()

        x, y = multiple_ureal([1, 2], [1, 1], 4)

        r = 0.5
        set_correlation(r, x, y)

        z = result(x + y)

        ar = persistence.Archive()

        ar.add(x=x, y=y, z=z)

        db = persistence.dumps_xml(ar)

        context._context = Context()
        ar = persistence.loads_xml(db)

        x1, y1, z1 = ar.extract('x', 'y', 'z')

        self.assertEqual(repr(x1), repr(x))
        self.assertEqual(repr(y1), repr(y))
        self.assertEqual(repr(z1), repr(z))

        self.assertEqual(get_correlation(x, y), r)

    def test_with_string5(self):
        """
        Complex
        """
        context._context = Context()

        x = ucomplex(1, [10, 2, 2, 10], 5)
        y = ucomplex(1 - 6j, [10, 2, 2, 10], 7)

        z = result(log(x * y))

        ar = persistence.Archive()

        ar.add(x=x, y=y, z=z)

        db = persistence.dumps_xml(ar)

        context._context = Context()
        ar = persistence.loads_xml(db)

        x1, y1, z1 = ar.extract('x', 'y', 'z')

        self.assertEqual(repr(x1), repr(x))
        self.assertEqual(repr(y1), repr(y))
        self.assertEqual(repr(z1), repr(z))

    def test_dumps_bytestring(self):
        ar = persistence.Archive()
        ar.add(x=ureal(1, 1))

        bytestring = persistence.dumps_xml(ar)
        self.assertTrue(isinstance(bytestring, bytes))
        persistence.loads_xml(bytestring)
        persistence.loads_xml(bytestring.decode())
        schema.assertValid(etree.fromstring(bytestring))

    def test_dumps_loads_unicode(self):
        ar = persistence.Archive()
        ar.add(x=ureal(1, 1))

        string = persistence.dumps_xml(ar, encoding='unicode')
        self.assertTrue(isinstance(string, str))
        schema.assertValid(etree.fromstring(string))

        for s in [string, string.encode()]:
            ar2 = persistence.loads_xml(s)
            x = ar2.extract('x')
            self.assertEqual(x.x, 1.0)
            self.assertEqual(x.u, 1.0)
            self.assertTrue(isinf(x.df))
            self.assertIsNone(x.label)

    def test_dump_load_filelike(self):
        ar = persistence.Archive()
        ar.add(x=ureal(1.1, 0.2, 3.4, label='a bc '))

        f = io.BytesIO()
        persistence.dump_xml(f, ar)
        f.seek(0)

        ar2 = persistence.load_xml(f)
        x = ar2.extract('x')
        self.assertEqual(x.x, 1.1)
        self.assertEqual(x.u, 0.2)
        self.assertEqual(x.df, 3.4)
        self.assertEqual(x.label, 'a bc ')

        schema.assertValid(etree.fromstring(f.getvalue()))
        f.close()

    def test_dump_load_filename(self):
        path = tempfile.mktemp()
        self.assertFalse(os.path.isfile(path))

        ar = persistence.Archive()
        ar.add(x=ureal(1, 1))

        persistence.dump_xml(path, ar)
        self.assertTrue(os.path.isfile(path))
        persistence.load_xml(path)
        os.remove(path)

    def test_dump_load_pathlike(self):
        from pathlib import Path

        path = Path(tempfile.mktemp())
        self.assertFalse(path.exists())

        ar = persistence.Archive()
        ar.add(x=ureal(1, 1))

        persistence.dump_xml(path, ar)
        self.assertTrue(path.exists())
        persistence.load_xml(path)
        os.remove(str(path))

    def test_labels(self):
        ar = persistence.Archive()

        x = ureal(10, 2, df=5.4, label='x')
        y = ureal(20, 2, df=9, label='y data')
        z = result(x + y, label='  z d a t a ')
        ar.add(x=x, y=y, z=z)

        mx, my, mz = multiple_ureal(
            [1, 2, 3], [0.1, 0.2, 0.3], 99,
            label_seq=['multi x', 'multi y', 'multi z'])
        mr = result(mx+my+mz, label='multi result')
        ar.add(mx=mx, my=my, mz=mz, mr=mr)

        cx = ucomplex(10-7j, 1, df=5.4, label='cx')
        cy = ucomplex(20+1j, (0.2, 0.5), df=9, label='cy data')
        cz = result(cx / cy, label=' c z d a t a ')
        ar.add(cx=cx, cy=cy, cz=cz)

        mca, mcb = multiple_ucomplex((2+3j, -1-2j), (0.4, 0.2), 6.3, label_seq=('foo', 'bar'))
        corr = (0.1, 0.2, 0.2, 0.1)
        set_correlation(corr, mca, mcb)
        mcc = result(mca/mcb, label='baz')
        ar.add(mca=mca, mcb=mcb, mcc=mcc)

        s = persistence.dumps_xml(ar)
        schema.assertValid(etree.fromstring(s))

        context._context = Context()
        ar1 = persistence.loads_xml(s)

        x1, y1, z1 = ar1.extract('x', 'y', 'z')
        self.assertEqual(repr(x1), repr(x))
        self.assertEqual(repr(y1), repr(y))
        self.assertEqual(repr(z1), repr(z))

        mx1, my1, mz1, mr1 = ar1.extract('mx', 'my', 'mz', 'mr')
        self.assertEqual(repr(mx1), repr(mx))
        self.assertEqual(repr(my1), repr(my))
        self.assertEqual(repr(mz1), repr(mz))
        self.assertEqual(repr(mr1), repr(mr))

        cx1, cy1, cz1 = ar1.extract('cx', 'cy', 'cz')
        self.assertEqual(repr(cx1), repr(cx))
        self.assertEqual(repr(cy1), repr(cy))
        self.assertEqual(repr(cz1), repr(cz))

        mca1, mcb1, mcc1 = ar1.extract('mca', 'mcb', 'mcc')
        self.assertTrue(equivalent_sequence(get_correlation(mca1, mcb1), corr))
        self.assertEqual(repr(mca1), repr(mca))
        self.assertEqual(repr(mcb1), repr(mcb))
        self.assertEqual(repr(mcc1), repr(mcc))

    def test_indent_0(self):
        context._context = Context(id=1)
        ar = persistence.Archive()
        ar.add(x=ureal(1, 1))
        s = persistence.dumps_xml(ar, indent=0)
        schema.assertValid(etree.fromstring(s))
        context._context = Context()
        expect = b"""<gtcArchive version="1.5.0" xmlns="https://measurement.govt.nz/gtc/xml">
<leafNodes>
<leafNode uid="(1, 1)">
<u>1.0</u>
<df>INF</df>
<label />
<independent>true</independent>
</leafNode>
</leafNodes>
<taggedReals>
<elementaryReal tag="x" uid="(1, 1)">
<value>1.0</value>
</elementaryReal>
</taggedReals>
<untaggedReals />
<taggedComplexes />
<intermediates />
</gtcArchive>"""
        self.assertEqual(s, expect)

    def test_indent_5(self):
        context._context = Context(id=1)
        ar = persistence.Archive()
        ar.add(x=ureal(1, 1))
        s = persistence.dumps_xml(ar, indent=5)
        schema.assertValid(etree.fromstring(s))
        context._context = Context()
        expect = b"""<gtcArchive version="1.5.0" xmlns="https://measurement.govt.nz/gtc/xml">
     <leafNodes>
          <leafNode uid="(1, 1)">
               <u>1.0</u>
               <df>INF</df>
               <label />
               <independent>true</independent>
          </leafNode>
     </leafNodes>
     <taggedReals>
          <elementaryReal tag="x" uid="(1, 1)">
               <value>1.0</value>
          </elementaryReal>
     </taggedReals>
     <untaggedReals />
     <taggedComplexes />
     <intermediates />
</gtcArchive>"""
        self.assertEqual(s, expect)

    def test_indent_invalid(self):
        # negative integer
        ar = persistence.Archive()
        ar.add(x=ureal(1, 1))
        self.assertRaises(ValueError, persistence.dumps_xml, ar, indent=-1)

        # not an integer
        ar2 = persistence.Archive()
        ar2.add(x=ureal(1, 1))
        self.assertRaises(TypeError, persistence.dumps_xml, ar2, indent=1.0)

    def test_invalid_root_tag(self):
        source = b'<msl><gtc>code</gtc></msl>'
        with self.assertRaises(ValueError) as err:
            persistence.loads_xml(source)
        self.assertEqual(
            str(err.exception),
            "Invalid root tag 'msl' for GTC Archive"
        )

        f = io.BytesIO(source)
        with self.assertRaises(ValueError) as err:
            persistence.load_xml(f)
        f.close()
        self.assertEqual(
            str(err.exception),
            "Invalid root tag 'msl' for GTC Archive"
        )

    def test_version_value(self):
        source = b'<gtcArchive/>'
        with self.assertRaises(ValueError) as err:
            persistence.loads_xml(source)
        self.assertEqual(
            str(err.exception),
            "Invalid XML Archive version 'UNKNOWN'"
        )

        f = io.BytesIO(source)
        with self.assertRaises(ValueError) as err:
            persistence.load_xml(f)
        f.close()
        self.assertEqual(
            str(err.exception),
            "Invalid XML Archive version 'UNKNOWN'"
        )

        source = b'<gtcArchive version="1.0"/>'
        with self.assertRaises(ValueError) as err:
            persistence.loads_xml(source)
        self.assertEqual(
            str(err.exception),
            "Invalid XML Archive version '1.0'"
        )

        f = io.BytesIO(source)
        with self.assertRaises(ValueError) as err:
            persistence.load_xml(f)
        f.close()
        self.assertEqual(
            str(err.exception),
            "Invalid XML Archive version '1.0'"
        )

    def test_dump_multiple_times(self):
        ar = persistence.Archive()
        ar.add(x=ureal(1, 1))
        source = persistence.dumps_xml(ar)
        schema.assertValid(etree.fromstring(source))

        for _ in range(10):
            self.assertEqual(source, persistence.dumps_xml(ar))

            with io.BytesIO() as f:
                persistence.dump_xml(f, ar)
                self.assertEqual(source, f.getvalue())

    def test_add_after_dump_raises(self):
        ar = persistence.Archive()
        ar.add(x=ureal(1, 1))
        persistence.dumps_xml(ar)

        with self.assertRaises(RuntimeError) as err:
            ar.add(y=ureal(1, 1))
        self.assertEqual(
            str(err.exception),
            'Archive cannot be added to'
        )

    def test_extract_after_dump_raises(self):
        ar = persistence.Archive()
        ar.add(x=ureal(1, 1))
        persistence.dumps_xml(ar)

        with self.assertRaises(RuntimeError) as err:
            ar.extract('x')
        self.assertEqual(
            str(err.exception),
            'Archive cannot be read'
        )

    def test_add_after_load_raises(self):
        ar = persistence.Archive()
        ar.add(x=ureal(1, 1))
        s = persistence.dumps_xml(ar)
        ar2 = persistence.loads_xml(s)

        with self.assertRaises(RuntimeError) as err:
            ar2.add(y=ureal(1, 1))
        self.assertEqual(
            str(err.exception),
            'Archive cannot be added to'
        )

    def test_dump_loaded_raises(self):
        ar = persistence.Archive()
        ar.add(x=ureal(1, 1))
        s = persistence.dumps_xml(ar)
        schema.assertValid(etree.fromstring(s))
        ar2 = persistence.loads_xml(s)

        with self.assertRaises(RuntimeError) as err:
            persistence.dumps_xml(ar2)
        self.assertEqual(
            str(err.exception),
            'Archive is not in the required state to be frozen'
        )

    def test_kwarg_method_raises(self):
        ar = persistence.Archive()
        ar.add(x=ureal(1, 1))

        msg = "Archive does not support method='text'"

        with self.assertRaises(ValueError) as err:
            persistence.dumps_xml(ar, method='text')
        self.assertEqual(str(err.exception), msg)

        with io.BytesIO() as f:
            with self.assertRaises(ValueError) as err:
                persistence.dump_xml(f, ar, method='text')
            self.assertEqual(str(err.exception), msg)

    def test_kwarg_method_xml_html(self):
        ar = persistence.Archive()
        ar.add(x=ureal(1, 1))

        for method in ['xml', 'html']:
            s = persistence.dumps_xml(ar, method=method)
            schema.assertValid(etree.fromstring(s))
            persistence.loads_xml(s)

            with io.BytesIO() as f:
                persistence.dump_xml(f, ar, method=method)
                schema.assertValid(etree.fromstring(f.getvalue()))
                f.seek(0)
                persistence.load_xml(f)

    def test_kwarg_default_namespace(self):
        ar = persistence.Archive()
        ar.add(x=ureal(1, 1))

        msg = "Archive uses a custom namespace, cannot set default_namespace='msl'"

        with self.assertRaises(ValueError) as err:
            persistence.dumps_xml(ar, default_namespace='msl')
        self.assertEqual(str(err.exception), msg)

        with io.BytesIO() as f:
            with self.assertRaises(ValueError) as err:
                persistence.dump_xml(f, ar, default_namespace='msl')
            self.assertEqual(str(err.exception), msg)

        # these are ok since they have a truthiness of False
        for ns in [None, '']:
            with io.BytesIO() as f:
                persistence.dump_xml(f, ar, default_namespace=ns)
                schema.assertValid(etree.fromstring(f.getvalue()))
                f.seek(0)
                persistence.load_xml(f)

    def test_kwarg_short_empty_elements(self):
        ar = persistence.Archive()
        ar.add(x=ureal(1, 1))

        for short in [True, False]:
            s = persistence.dumps_xml(ar, short_empty_elements=short)
            schema.assertValid(etree.fromstring(s))
            persistence.loads_xml(s)

            with io.BytesIO() as f:
                persistence.dump_xml(f, ar, short_empty_elements=short)
                schema.assertValid(etree.fromstring(f.getvalue()))
                f.seek(0)
                persistence.load_xml(f)

    def test_namespace_prefix_invalid(self):
        ar = persistence.Archive()
        ar.add(x=ureal(1, 1))

        with self.assertRaises(ValueError) as err:
            persistence.dumps_xml(ar, prefix='XmlNS')
        self.assertEqual(
            str(err.exception),
            "An XML namespace prefix should not start with 'xml', got prefix='XmlNS'"
        )

        with self.assertRaises(ValueError) as err:
            persistence.dumps_xml(ar, prefix='m:sl')
        self.assertEqual(
            str(err.exception),
            "An XML namespace prefix cannot contain a colon, got prefix='m:sl'"
        )

    def test_namespace_prefix(self):
        context._context = Context(id=1)

        ar = persistence.Archive()

        w = ureal(1, 1)
        x = ureal(2, 1)
        y = result(w + x)
        z = result(x * y)

        ar.add(w=w, x=x, y=y, z=z)

        x1 = ureal(1, 1, 3, label='x1')
        y1 = ureal(2, 1, 4, label='y1')
        z1 = result(x1 + y1)

        ar.add(z1=z1)

        x2 = ureal(1, 1, independent=False)
        y2 = ureal(2, 1, independent=False)

        r = 0.5
        set_correlation(r, x2, y2)

        z2 = result(x2 + y2)

        ar.add(x2=x2, y2=y2, z2=z2)

        x3, y3 = multiple_ureal([1, 2], [1, 1], 4)

        r = 0.5
        set_correlation(r, x3, y3)

        z3 = result(x3 + y3)

        ar.add(x3=x3, y3=y3, z3=z3)

        x4 = ucomplex(1, [10, 2, 2, 10], 5)
        y4 = ucomplex(1 - 6j, [10, 2, 2, 10], 7)

        z4 = result(log(x4 * y4))

        ar.add(x4=x4, y4=y4, z4=z4)

        s = persistence.dumps_xml(ar, indent=2, prefix='msl')
        schema.assertValid(etree.fromstring(s))

        context._context = Context()

        expect = b"""<msl:gtcArchive version="1.5.0" xmlns:msl="https://measurement.govt.nz/gtc/xml">
  <msl:leafNodes>
    <msl:leafNode uid="(1, 1)">
      <msl:u>1.0</msl:u>
      <msl:df>INF</msl:df>
      <msl:label />
      <msl:independent>true</msl:independent>
    </msl:leafNode>"""

        self.assertTrue(s.startswith(expect))

        ar = persistence.loads_xml(s)

        w, x, y, z = ar.extract('w', 'x', 'y', 'z')

        _w = ureal(1, 1)
        _x = ureal(2, 1)
        _y = _w + _x
        _z = _x * _y
        self.assertEqual(repr(w), repr(_w))
        self.assertEqual(repr(x), repr(_x))
        self.assertEqual(repr(y), repr(_y))
        self.assertEqual(repr(z), repr(_z))
        self.assertEqual(component(z, w), 2)
        self.assertEqual(component(z, x), 5)
        self.assertEqual(component(z, y), 2 * uncertainty(_w + _x))

        z1 = ar.extract('z1')
        _z1 = ureal(1, 1, 3) + ureal(2, 1, 4)
        self.assertEqual(repr(z1), repr(_z1))

        x1, y1, z1 = ar.extract('x2', 'y2', 'z2')

        x = ureal(1, 1, independent=False)
        y = ureal(2, 1, independent=False)
        r = 0.5
        set_correlation(r, x, y)
        z = x + y

        self.assertEqual(repr(x1), repr(x))
        self.assertEqual(repr(y1), repr(y))
        self.assertEqual(repr(z1), repr(z))

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
