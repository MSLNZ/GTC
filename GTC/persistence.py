"""
Functions
---------    

    Functions for storing and retrieving archive files using JSON format are

        * :func:`dump_json`
        * :func:`load_json`

    Functions for storing and retrieving an archive as a JSON-formatted string are
    
        * :func:`dumps_json`
        * :func:`loads_json`

    Functions for storing and retrieving archive files using XML format are

        * :func:`dump_xml`
        * :func:`load_xml`

    Functions for storing and retrieving an archive as an XML-formatted string are

        * :func:`dumps_xml`
        * :func:`loads_xml`

Module contents
---------------

"""
import json
import xml.etree.cElementTree as ElementTree

from GTC.archive import Archive

from GTC.json_format import (
    JSONArchiveEncoder,
    json_to_archive,
)
from GTC.xml_format import (
    archive_to_xml,
    xml_to_archive,
)

__all__ = (
    'Archive',
    'dump_json',
    'load_json',
    'dumps_json',
    'loads_json',
    'dump_xml',
    'load_xml',
    'dumps_xml',
    'loads_xml',
)


#------------------------------------------------------------------     
def dumps_json(ar,**kw):
    """
    Convert an archive to a JSON string  

    :arg ar: an :class:`Archive` object
    
    Keyword arguments will be passed to :func:`json.dumps()`

    .. versionadded:: 1.3.0
    """
    ar._freeze()
    s = json.dumps(ar, cls=JSONArchiveEncoder,**kw )
    
    return s

#------------------------------------------------------------------     
def loads_json(s,**kw):
    """
    Return an archive object by converting a JSON string  

    :arg s: a string created by :func:`dumps_json`
    
    Keyword arguments will be passed to :func:`json.loads()`
    
    .. versionadded:: 1.3.0
    """
    ar = json.loads(s,object_hook=json_to_archive,**kw)
    ar._thaw()

    return ar
    
#------------------------------------------------------------------     
def dump_json(file,ar,**kw):
    """Save an archive in a file in JSON format

    :arg file:  a file object opened in text mode (with 'w')
                
    :arg ar: an :class:`Archive` object
      
    Keyword arguments will be passed to :func:`json.dump()`

    Only one archive can be saved in a file.
    
    .. versionadded:: 1.3.0
    """
    ar._freeze()
    s = json.dump(ar, file, cls=JSONArchiveEncoder, **kw )
    
    return s

#------------------------------------------------------------------     
def load_json(file,**kw):
    """Load an archive from a file

    :arg file: a file created by :func:`dump_json`
    
    Keyword arguments will be passed to :func:`json.load()`

    .. versionadded:: 1.3.0
    """
    s = file.read()
    ar = loads_json(s, **kw)
    
    return ar


def dumps_xml(ar, indent=None, prefix=None, **kw):
    """Convert an archive to an XML document bytestring (or string).

    :arg ar: an :class:`Archive` object.
    :arg indent: the indentation to apply between XML elements so that the
        XML document is in a pretty-printed format. The `indent` value must
        be a non-negative integer.
    :type indent: int or None
    :arg prefix: The prefix to use for the XML namespace.
    :type prefix: str or None

    Keyword arguments will be passed to
    :func:`ElementTree.tostring() <xml.etree.ElementTree.tostring()>`.

    The return type, :class:`bytes` or :class:`str`, depends on whether
    an `encoding` keyword argument is specified and what its value is. The
    default return type is :class:`bytes`.

    .. versionadded:: 1.5.0
    """
    _check_xml_kwargs(**kw)
    element = archive_to_xml(ar, indent=indent, prefix=prefix)
    return ElementTree.tostring(element, **kw)


def loads_xml(s):
    """Return an :class:`Archive` object by converting an XML string.

    :arg s: a string created by :func:`dumps_xml`.
    :type s: bytes or str

    .. versionadded:: 1.5.0
    """
    return xml_to_archive(ElementTree.XML(s))


# ------------------------------------------------------------------
def dump_xml(file, ar, indent=None, prefix=None, **kw):
    """Save an archive in a file in XML format.

    :arg file: a file name or a file-like object that can be written to.
    :arg ar: an :class:`Archive` object.
    :arg indent: the indentation to apply between XML elements so that the
        XML document is in a pretty-printed format. The `indent` value must
        be a non-negative integer.
    :type indent: int or None
    :arg prefix: The prefix to use for the XML namespace.
    :type prefix: str or None

    Keyword arguments will be passed to
    :meth:`ElementTree.write() <xml.etree.ElementTree.ElementTree.write()>`.

    Only one archive can be saved in a file.

    .. versionadded:: 1.5.0
    """
    _check_xml_kwargs(**kw)
    element = archive_to_xml(ar, indent=indent, prefix=prefix)
    ElementTree.ElementTree(element).write(file, **kw)


# ------------------------------------------------------------------
def load_xml(file):
    """Load an :class:`Archive` from a file in XML format.

    :arg file: a file name or a file-like object that can be read.

    .. versionadded:: 1.5.0
    """
    tree = ElementTree.ElementTree(file=file)
    return xml_to_archive(tree.getroot())


def _check_xml_kwargs(**kwargs):
    if kwargs.get('method') == 'text':
        raise ValueError("Archive does not support method='text'")

    ns = kwargs.get('default_namespace')
    if ns:
        raise ValueError(
            f'Archive uses a custom namespace, '
            f'cannot set default_namespace={ns!r}'
        )


#============================================================================
if __name__ == "__main__":
    import doctest
    from GTC import *  
    doctest.testmod( optionflags=doctest.NORMALIZE_WHITESPACE )
