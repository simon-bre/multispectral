import re
import os
from statistics import mode
import warnings

class Layer:
    """ Represents an image that is part of a Frame """

    def __init__(self, name, file):
        """
        :param name: just some descriptive string
        :param file: path to image file
        """
        self.name = name
        self.file = file


class Frame:
    """
    A Frame is a set of Layers (images) of the same size that depict the exact same scene (eg. a Multispectral image)
    """

    def __init__(self, root_dir, name='unnamed_frame', regex='', group_framename=-1, group_layername=-1):
        """
        :param root_dir: root directory (preferably full path) of the frame
        :param name: a descriptive name of the frame. will be overwritten if group_framename is specified
        :param regex: regular expression used to search for layers (=image files) in the root_dir
                        it can contain groups "(<some expression)" that can be used to extract the base name of the
                        frame and the identifier for the layer.
                        example: '(.*)_.*_(.*).tif' matches 'abook-42r_something-irrelevant_365nm.tif' and captures
                        groups 'abook-42r' and '365nm'.
        :param group_framename: index of regex group containing the frame name
                                for the example above: use '1' to select 'abook-42r'
        :param group_layername: index of regex group containing the layer identifier
                                for the example above: use '2' to select '365nm'
        """

        #empty frame
        self.root_dir = root_dir
        self.name = name
        self.layers = []

        #collect layers by parsing root_dir
        if root_dir is not None:
            self.layers, framename_candidates = self.__collect_layers(root_dir, regex, group_framename, group_layername)

            voted_framename =self.__vote_for_framename(framename_candidates)
            self.name = name if voted_framename is None else voted_framename



    def __collect_layers(self, root_dir, regex, group_framename, group_layername):
        """
        Recursively parses the root_dir and collects all files matching the regex. Parameters are passed unchanged
        from __init__
        :param root_dir:
        :param regex:
        :param group_framename:
        :param group_layername:
        :return: list of collected Layer(s), list of possible layer names
        """

        layers = []
        framename_candidates = []
        for d, _, files in os.walk(root_dir):
            i = 0
            for f in files:
                result = re.search(regex, f)
                if result is None:
                    continue
                if group_framename >= 0:
                    framename_candidates.append(result.group(group_framename))
                if group_layername >= 0:
                    layername = result.group(group_layername)
                else:
                    layername = "layer_%d" % i
                i += 1
                layers.append(Layer(name=layername, file=os.path.abspath(os.path.join(d, f))))

        if layers == []:
            raise Exception('No layers matching "%s" found in %s' % (regex, root_dir))
        return layers, framename_candidates


    def __vote_for_framename(self, candidates):
        """In case different base names are encountered while parsing the layers, a voting takes place."""
        try:
            return mode(candidates)
        except:
            return None

    def append(self, other):
        """Appends the layers of another frame, or single layer. Name and root dir stay the same."""
        if type(other) is Frame:
            self.layers.extend(other.layers)
        elif type(other) is Layer:
            self.layers.append(other)
        else:
            warnings.warn("Trying to append somethings that we don't know.")
