"""
This module handles conversion of an Archive object to XML format
and then restoration of an Archive from XML.
"""
from ast import literal_eval
from math import isinf
from xml.etree.cElementTree import (
    Element,
    SubElement,
)

from GTC.archive import (
    Archive,
    LeafNode,
    ElementaryReal,
    IntermediateReal,
    Complex,
    PY2,
)
from GTC.nodes import Leaf
from GTC.vector import Vector

XMLNS = 'https://measurement.govt.nz/gtc/xml'


def _py38indent(tree, space="  ", level=0):
    # TODO An xml.etree.ElementTree.indent() function was added in Python 3.9.
    #  This function is basically a copy and paste of the implementation from
    #  cpython/Lib/xml/etree/ElementTree.py apart from a few "if" checks at
    #  the beginning of the builtin function. The builtin indent() function
    #  may be used when GTC no longer supports Python 3.8 (and below) and this
    #  custom _py38indent() function should be deleted.

    # Reduce the memory consumption by reusing indentation strings.
    indentations = ["\n" + level * space]

    def _indent_children(elem, level):  # noqa
        # Start a new indentation level for the first child.
        child_level = level + 1
        try:
            child_indentation = indentations[child_level]
        except IndexError:
            child_indentation = indentations[level] + space
            indentations.append(child_indentation)

        if not elem.text or not elem.text.strip():
            elem.text = child_indentation

        for child in elem:
            if len(child):
                _indent_children(child, child_level)
            if not child.tail or not child.tail.strip():
                child.tail = child_indentation

        # Dedent after the last child by overwriting the previous indentation.
        if not child.tail.strip():  # noqa
            child.tail = indentations[level]

    _indent_children(tree, 0)


def _py27uid(iterable):
    # TODO In Python 2, the letter "L" is appended to all long integers
    #  when repr() is applied to the object but not when str() is applied.
    #  In Python 3, the concept of long integers does not exist. Since a
    #  Context currently uses long integers for the id's, the "L" must be
    #  suppressed when Python 2.7 is used to convert an Archive to XML so
    #  that the XML document can be validated against the XML Schema (don't
    #  want to allow an "L" in the Schema regex). Once support for Python
    #  2.7 isdropped, this _py27uid() function should be deleted and the
    #  uid (as a tuple of ints) can be directly used as the attribute
    #  value of an XML Element.
    if PY2:
        return '(' + ', '.join(map(str, iterable)) + ')'
    return iterable


def _find(parent, name):
    # Find and return the first matching sub-element
    return parent.find(f'{{{XMLNS}}}{name}')


def _float(parent, name):
    # Convert an element's text attribute to a float
    return float(_find(parent, name).text)


def archive_to_xml(archive, indent=None, prefix=None):
    """Convert an Archive to an XML element.

    :param archive: The Archive to convert.
    :type archive: :class:`GTC.archive.Archive`
    :param indent: The indentation to apply between XML elements so that the
        XML document is in a pretty-printed format. The `indent` value must
        be a non-negative integer.
    :type indent: int or None
    :arg prefix: The prefix to use for the XML namespace.
    :type prefix: str or None

    :returns: The XML element.
    :rtype: :class:`xml.etree.ElementTree.Element`
    """
    def add_components(parent, name, items):
        # Add Vector components to the parent element
        elem = SubElement(parent, f'{pre}{name}')
        for k, v in items():
            SubElement(elem, f'{pre}component', uid=_py27uid(k)).text = str(v)

    def add_real(parent, tag, real):
        # Add an ElementaryReal or IntermediateReal to the parent element
        if isinstance(real, ElementaryReal):
            er = SubElement(parent, f'{pre}elementaryReal', tag=tag, uid=_py27uid(real.uid))
            SubElement(er, f'{pre}value').text = str(real.x)
        elif isinstance(real, IntermediateReal):
            ir = SubElement(parent, f'{pre}intermediateReal', tag=tag, uid=_py27uid(real.uid))
            SubElement(ir, f'{pre}value').text = str(real.value)
            SubElement(ir, f'{pre}label').text = real.label
            add_components(ir, 'uComponents', real.u_components.items)
            add_components(ir, 'dComponents', real.d_components.items)
            add_components(ir, 'iComponents', real.i_components.items)
        else:
            assert False, 'not ElementaryReal or IntermediateReal'

    def normalise_df(dof):
        # Ensures that the DoF is valid XML text for type xsd:double
        return 'INF' if isinf(dof) else str(dof)

    archive._freeze()

    if PY2:
        leaf_nodes_items = archive._leaf_nodes.iteritems
        tagged_real_items = archive._tagged_real.iteritems
        tagged_complex_items = archive._tagged_complex.iteritems
        untagged_real_items = archive._untagged_real.iteritems
        intermediate_uids_items = archive._intermediate_uids.iteritems
    else:
        leaf_nodes_items = archive._leaf_nodes.items
        tagged_real_items = archive._tagged_real.items
        tagged_complex_items = archive._tagged_complex.items
        untagged_real_items = archive._untagged_real.items
        intermediate_uids_items = archive._intermediate_uids.items

    if prefix:
        if prefix.lower().startswith('xml'):
            # From W3C -> Namespaces in XML 1.0 (Third Edition)
            # https://www.w3.org/TR/xml-names/#ns-using
            #   "All other prefixes beginning with the three-letter sequence
            #    x, m, l, in any case combination, are reserved."
            raise ValueError(
                f"An XML namespace prefix should not start with 'xml', "
                f"got prefix={prefix!r}")
        if ':' in prefix:
            raise ValueError(
                f"An XML namespace prefix cannot contain a colon, "
                f"got prefix={prefix!r}")

        xmlns = (f'xmlns:{prefix}', XMLNS)
        pre = f'{prefix}:'
    else:
        xmlns = ('xmlns', XMLNS)
        pre = ''

    root = Element(f'{pre}gtcArchive', version='1.5.0')
    root.set(*xmlns)

    leaf_nodes = SubElement(root, f'{pre}leafNodes')
    for uid, ln in leaf_nodes_items():
        assert uid == ln.uid, f'LeafNode(uid={ln.uid}) != {uid}'
        leaf_node = SubElement(leaf_nodes, f'{pre}leafNode', uid=_py27uid(uid))
        SubElement(leaf_node, f'{pre}u').text = str(ln.u)
        SubElement(leaf_node, f'{pre}df').text = normalise_df(ln.df)
        SubElement(leaf_node, f'{pre}label').text = ln.label
        SubElement(leaf_node, f'{pre}independent').text = 'true' if ln.independent else 'false'
        if hasattr(ln, 'complex'):
            c = SubElement(leaf_node, f'{pre}complex')
            SubElement(c, f'{pre}real', uid=_py27uid(ln.complex[0]))
            SubElement(c, f'{pre}imag', uid=_py27uid(ln.complex[1]))
        if hasattr(ln, 'correlation'):
            c = SubElement(leaf_node, f'{pre}correlations')
            for cid, value in ln.correlation:
                SubElement(c, f'{pre}correlation', uid=_py27uid(cid)).text = str(value)
        if hasattr(ln, 'ensemble'):
            e = SubElement(leaf_node, f'{pre}ensemble')
            for eid in ln.ensemble:
                SubElement(e, f'{pre}node', uid=_py27uid(eid))

    tagged_reals = SubElement(root, f'{pre}taggedReals')
    for key, value in tagged_real_items():
        add_real(tagged_reals, key, value)

    untagged_reals = SubElement(root, f'{pre}untaggedReals')
    for key, value in untagged_real_items():
        add_real(untagged_reals, key, value)

    tagged_complexes = SubElement(root, f'{pre}taggedComplexes')
    for key, value in tagged_complex_items():
        c = SubElement(tagged_complexes, f'{pre}complex', tag=key)
        SubElement(c, f'{pre}label').text = value.label

    intermediates = SubElement(root, f'{pre}intermediates')
    for key, value in intermediate_uids_items():
        label, u, df = value
        inter = SubElement(intermediates, f'{pre}intermediate', uid=_py27uid(key))
        SubElement(inter, f'{pre}label').text = label
        SubElement(inter, f'{pre}u').text = str(u)
        SubElement(inter, f'{pre}df').text = normalise_df(df)

    if indent is not None:
        if indent < 0:
            raise ValueError(f'XML indentation must be >= 0, got {indent}')
        _py38indent(root, space=' ' * indent)

    return root


def xml_to_archive(element):
    """Convert an XML element to an Archive.

    :param element: The XML element to convert.
    :type element: :class:`xml.etree.ElementTree.Element`

    :returns: The Archive.
    :rtype: :class:`GTC.archive.Archive`
    """
    if not element.tag.endswith('gtcArchive'):
        raise ValueError(f'Invalid root tag {element.tag!r} for GTC Archive')

    version = element.get('version', 'UNKNOWN')
    if version == '1.5.0':
        return _v150_to_archive(element)

    raise ValueError(f'Invalid XML Archive version {version!r}')


def _v150_to_archive(root):
    # Convert XML version 1.5.0 to an Archive
    def convert_leaf_node(elem):
        # Returns: tuple(uid:tuple, LeafNode)
        uid = literal_eval(elem.get('uid'))
        leaf = Leaf(
            uid=uid,
            label=_find(elem, 'label').text,
            u=_float(elem, 'u'),
            df=_float(elem, 'df'),
            independent=_find(elem, 'independent').text == 'true',
        )
        cmplx = _find(elem, 'complex')
        if cmplx is not None:
            leaf.complex = (
                literal_eval(_find(cmplx, 'real').get('uid')),
                literal_eval(_find(cmplx, 'imag').get('uid')),
            )
        correlations = _find(elem, 'correlations')
        if correlations is not None:
            leaf.correlation = {
                literal_eval(el.get('uid')): float(el.text)
                for el in correlations}
        ensemble = _find(elem, 'ensemble')
        if ensemble is not None:
            leaf.ensemble = frozenset(
                literal_eval(el.get('uid')) for el in ensemble)
        return uid, LeafNode(leaf)

    def convert_components(elem, name):
        # Returns: Vector
        index, value = [], []
        for component in _find(elem, name):
            index.append(literal_eval(component.get('uid')))
            value.append(float(component.text))
        return Vector(index=index, value=value)

    def convert_real(elem):
        # Returns: tuple(tag:str, ElementaryReal or IntermediateReal)
        if elem.tag.endswith('elementaryReal'):
            er = ElementaryReal(
                x=_float(elem, 'value'),
                uid=literal_eval(elem.get('uid')))
            return elem.get('tag'), er
        if elem.tag.endswith('intermediateReal'):
            ir = IntermediateReal(
                value=_float(elem, 'value'),
                u_components=convert_components(elem, 'uComponents'),
                d_components=convert_components(elem, 'dComponents'),
                i_components=convert_components(elem, 'iComponents'),
                label=_find(elem, 'label').text,
                uid=literal_eval(elem.get('uid')))
            return elem.get('tag'), ir
        assert False, 'not elementaryReal or intermediateReal'

    def convert_complex(elem):
        # Returns: tuple(tag:str, Complex)
        t = elem.get('tag')
        c = Complex(n_re=f'{t}_re',
                    n_im=f'{t}_im',
                    label=_find(elem, 'label').text)
        return t, c

    def convert_intermediate(elem):
        # Returns: tuple(uid:tuple, tuple(label, u, df))
        data = (_find(elem, 'label').text,
                _float(elem, 'u'),
                _float(elem, 'df'))
        return literal_eval(elem.get('uid')), data

    archive = Archive()  
    archive._dump = archive._ready = False

    # Load the data
    archive._leaf_nodes = dict(
        convert_leaf_node(element)
        for element in _find(root, 'leafNodes')
    )

    archive._tagged_real = dict(
        convert_real(element)
        for element in _find(root, 'taggedReals')
    )

    archive._tagged_complex = dict(
        convert_complex(element)
        for element in _find(root, 'taggedComplexes')
    )

    archive._untagged_real = dict(
        convert_real(element)
        for element in _find(root, 'untaggedReals')
    )

    archive._intermediate_uids = dict(
        convert_intermediate(element)
        for element in _find(root, 'intermediates')
    )

    archive._thaw()
    return archive
