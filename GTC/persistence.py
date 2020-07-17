"""
Functions
---------    

    Functions for storing and retrieving pickled archive files are
    
        * :func:`load`
        * :func:`dump`
        
    Functions for storing and retrieving pickled archive strings are

        * :func:`dumps`
        * :func:`loads`
        
    Functions for storing and retrieving JSON-formatted archive strings are
    
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
from archive import Archive
from json_format import JSONArchiveEncoder, json_to_archive

__all__ = (
    'Archive',
    'load',
    'dump',
    'dumps',
    'loads',
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
    pickle.dump(ar,file,protocol=pickle.HIGHEST_PROTOCOL)

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

    Possible values for ``protocol`` are described in the 
    Python documentation for the 'pickle' module.

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
    
    """
    ar._freeze()
    s = json.dumps(ar, cls=JSONArchiveEncoder,**kw )
    
    return s

#------------------------------------------------------------------     
def loads_json(s,**kw):
    """
    Return an archive object by converting a JSON string  

    :arg s: a string created by :func:`dumps_json`
    
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
      
    Only one archives can be saved in the same file.
    
    """
    ar._freeze()
    s = json.dump(ar, file, cls=JSONArchiveEncoder, **kw )
    
    return s

#------------------------------------------------------------------     
def load_json(file,**kw):
    """
    Return an archive object by converting a JSON string  

    :arg s: a string created by :func:`dumps_json`
    
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
