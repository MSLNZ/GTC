"""
Functions
---------    

    Functions for storing and retrieving archive files using Python pickle format are
    
        * :func:`dump`
        * :func:`load`

    Functions for storing and retrieving pickled archive strings are

        * :func:`dumps`
        * :func:`loads`

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
import re
import json
import xml.etree.cElementTree as ElementTree

try:
    import cPickle as pickle  # Python 2
    PY2 = True
except ImportError:
    import pickle
    PY2 = False

from GTC import archive_old

from GTC.archive import Archive

# Support for legacy format will be dropped in GTC 2
from GTC import json_format_old

from GTC.json_format import (
    JSONArchiveEncoder,
    json_to_archive,
    JSON_SCHEMA,
)
from GTC.xml_format import (
    archive_to_xml,
    xml_to_archive,
)

__all__ = (
    'Archive',
    'load',
    'dump',
    'dumps',
    'loads',
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
def dump(file,ar):
    """Save an archive in a file

    :arg file:  a file object opened in binary mode (with 'wb')
                
    :arg ar: an :class:`Archive` object
      
    Several archives can be saved in the same file 
    by repeated use of this function.
    
    """
    ar._freeze()

# About pickle protocols (from 3.8 docs, abridged)
# 0 is the original 'human-readable' protocol and is backwards compatible with earlier versions of Python.
# 1 is an old binary format which is also compatible with earlier versions of Python.
# 2 was introduced in Python 2.3. It provides much more efficient pickling of new-style classes. 
# 3 was added in Python 3.0. It has explicit support for bytes objects and cannot be unpickled by Python 2.x. 
#    This was the default protocol in Python 3.0 - 3.7.
# 4 was added in Python 3.4. It adds support for very large objects, pickling more kinds of objects, and some data format optimizations. 
#    It is the default protocol starting with Python 3.8. 
# 5 was added in Python 3.8. It adds support for out-of-band data and speedup for in-band data. 

    pickle.dump(ar,file,protocol=2)     # Change to 3 when GTC no longer supports Python 2.7

#------------------------------------------------------------------     
def load(file):
    """Load an archive from a file

    :arg file:  a file object opened in binary mode (with 'rb')

    Several archives can be extracted from 
    the same file by repeatedly calling this function.
    
    """
    ar = pickle.load(file)
    
    # Pickle may return a new-style Archive when unpickling a file 
    # containing the old-style class (pickle takes any Archive definition). 
    if hasattr(ar,"_tagged"):
        file.seek(0)
        
        old = archive_old.load(file)
        old._dump = False
        old._ready = False     
        old._thaw()
        
        ar = Archive.from_old_archive(old)
    else:
        ar._dump = False
        ar._ready = False     
        ar._thaw()
    
    return ar

#------------------------------------------------------------------     
def dumps(ar,protocol=pickle.HIGHEST_PROTOCOL):
    """
    Save an archive pickled in a string  

    :arg ar: an :class:`Archive` object
    :arg protocol: encoding type 

    Possible values for :ref:`protocol <pickle-protocols>` are described in the
    Python documentation for the :mod:`pickle` module.

    ``protocol=0`` creates an ASCII string, but note
    that many (special) linefeed characters are embedded.
    
    """
    # Can save one of these strings in a single binary file,
    # using write(), when protocol=pickle.HIGHEST_PROTOCOL is used. 
    # A corresponding read() is required to extract the string. 
    # Alternatively, when protocol=0 is used a text file can be 
    # used, but again write() and read() have to be used, 
    # because otherwise the embedded `\n` characters are 
    # interpreted incorrectly.
    
    ar._freeze()
    s = pickle.dumps(ar,protocol)
    
    return s
    
#------------------------------------------------------------------     
def loads(s):
    """
    Return an archive object from a pickled string 

    :arg s: a string created by :func:`dumps`
    
    """
    ar = pickle.loads(s)
    ar._dump = False
    ar._ready = False     
    ar._thaw()
    
    return ar

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
    # Support for legacy JSON format will be dropped in GTC 2
    pattern = r'"version": "{}"'.format(
        re.sub(r'\.', r'\.',JSON_SCHEMA)
    )
    if re.search(pattern, s):
        ar = json.loads(s,object_hook=json_to_archive,**kw)    
        # ar._dump = False
        # ar._ready = False     
        ar._thaw()
    else:
        old = json_format_old.json.loads(
            s,
            object_hook=json_format_old.json_to_archive,
            **kw
        )    
        # old._dump = False
        # old._ready = False     
        old._thaw()
        ar = Archive.from_old_archive(old)
    
    
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
    ar = loads_json(s,**kw)
    
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
            'Archive uses a custom namespace, '
            'cannot set default_namespace={!r}'.format(ns)
        )


#============================================================================
if __name__ == "__main__":
    import doctest
    from GTC import *  
    doctest.testmod( optionflags=doctest.NORMALIZE_WHITESPACE )
