# -*- coding: UTF-8 -*-
"""A lazy manager for the experimetn results.
"""
import os
import re
import gzip
import cPickle

def depth_first_filter(cache, key):
    """Depth-first filtered traversal of a hierarchical
    dictionary. Requires that the key be padded.
    """
    # Recursion depth is controlled by the length of the key.
    #  pad the key with entire slices to limit the depth.
    if not isinstance(key, (tuple, list)):
        key = tuple([key])

    # Use entire slice, if the key has been exhausted.
    key_ = key[0] if key else slice(None)
    if isinstance(key_, slice):
        if key_.stop or key_.start or key_.step:
            raise TypeError("""Only entire slices are supported.""")
        generator_ = cache.iteritems()
    else:
        if key_ not in cache:
            raise StopIteration
        generator_ = iter([(key_, cache[key_])])

    # Depth-first traversal
    for key_, value_ in generator_:
        if len(key) > 1:
            for k, v in depth_first_filter(value_, key[1:]):
                yield ((key_,) + k, v)
        else:
            yield ((key_,), value_)

class Hierarchy(object):
    """A simple container for hierarchy."""
    def __init__(self, depth=5):
        self.depth = depth
        self.base = dict()
    
    def __call__(self, key=None):
        if not isinstance(key, (tuple, list)):
            if key is None:
                key = (Ellipsis,)
            else:
                key = tuple([key])

        # pad the key unitl full depth
        ell_index_ = [j for j, member_ in enumerate(key)
                      if member_ is Ellipsis]
        if len(ell_index_) > 1:
            raise TypeError("""Too many ellipses.""")

        if ell_index_:
            left_, right_ = key[:ell_index_[0]], key[ell_index_[0] + 1:]
        else:
            left_, right_ = key, tuple()

        pad_ = self.depth - len(left_) - len(right_)
        if pad_ < 0:
            raise KeyError("""Invalid key (`%s`)"""%("`, `".join(
                str(member_) for member_ in key),))

        key_ = left_ + tuple([slice(None)]*pad_) + right_
        return key_, depth_first_filter(self.base, key_)

    def __getitem__(self, key):
        key_, iterator_ = self.__call__(key)
        result = list(iterator_)
        if any(isinstance(member_, slice) for member_ in key_):
            return result

        if len(result) > 0:
            return result[0]

        raise KeyError(str(key_))

    def __iter__(self):
        return self.__call__(slice(None))[1]

    def __setitem__(self, key, value):
        if len(key) != self.depth:
            raise KeyError("""Invalid key depth.""")

        base_ = self.base
        for key_ in key[:-1]:
            if key_ not in base_:
                base_[key_] = dict()
            base_ = base_[key_]
        base_[key[-1]] = value

    def __contains__(self, key):
        """Checks if an `item` is in the cache.
        """
        if len(key) != self.depth:
            return False

        base_ = self.base
        for key_ in key:
            if key_ not in base_:
                return False
            base_ = base_[key_]
        return True

class ExperimentManager(object):
    """docstring for ExperimentManager"""
    def __init__(self, name_format):
        """A simple lazy manager for the experiment results.
        """
        self.format = name_format

        # Initialize the keys and cache
        self.keys_ = [tup_[0] for tup_ in
                      sorted(self.format.groupindex.iteritems(),
                             key=lambda tup_: tup_[1])]
        self.experiments_ = Hierarchy(depth=len(self.keys_))
        self.cache_ = dict()

    def update(self, path):
        """Scans the directory of the experiment."""
        # Add files to the cache
        for base_ in os.listdir(path):
            filename_ = os.path.join(path, base_)
            key_ = self.check(filename_)
            if not key_:
                continue
            self.experiments_[key_] = filename_

    def check(self, filename):
        """Check if the `filename` conforms to the sepecified format."""
        matches_ = self.format.match(os.path.basename(filename))
        if matches_ is None:
            return None
        return tuple(matches_.group(key_) for key_ in self.keys_)
    
    def load(self, filename):
        """Loads the results of an experiment.
        """
        info = self.check(filename)
        if info is None:
            raise TypeError("""Cannot identify file: `%s`."""
                            %(os.path.basename(filename),))
        with gzip.open(filename, "rb") as fin:
            start, finish, seeds, results = cPickle.load(fin)
        return info, start, finish, seeds, results

    def __contains__(self, key):
        return key in self.experiments_

    def __getitem__(self, key):
        full_key_, iterator_ = self.experiments_(key)
        
        list_ = list()
        for key_, filename_ in iterator_:
            if filename_ not in self.cache_:
                self.cache_[filename_] = self.load(filename_)
            list_.append(self.cache_[filename_])
    
        # Return a list of items if the key has a slice
        if any(isinstance(member_, slice) for member_ in full_key_):
            return list_
        
        if len(list_) > 0:
            return list_[0]

        raise KeyError(str(key))
