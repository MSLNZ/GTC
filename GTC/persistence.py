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

Module contents
---------------

"""
import json
try:
    import cPickle as pickle  # Python 2
    PY2 = True
except ImportError:
    import pickle
    PY2 = False

from GTC import context
from GTC.archive import Archive
from GTC.json_format import (
    JSONArchiveEncoder,
    json_to_archive,
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
    ar.context = context._context
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
    ar.context = context._context
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
    ar = json.loads(s,object_hook=json_to_archive,**kw)    
    ar.context = context._context
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
    ar = json.load(file, object_hook=json_to_archive,**kw)    
    ar.context = context._context
    ar._thaw()
    
    return ar
    
#============================================================================    
if __name__ == "__main__":
    import doctest
    from GTC import *  
    doctest.testmod( optionflags=doctest.NORMALIZE_WHITESPACE )
